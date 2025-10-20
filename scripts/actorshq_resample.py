from __future__ import annotations

"""
Pipeline to resample ActorsHQ sequences with synthetic camera rigs and prepare
data for 4DGaussians multiple view training.

High-level stages:
1. Load ActorsHQ calibration and scene metadata to build a `SceneInfo` snapshot
   compatible with the gs7 camera generation utilities.
2. Invoke gs7's rig constructors (lightfield grids, stereo pairs, iPhone triple
   cameras, etc.) to synthesize new virtual cameras and export their calibration.
3. Spawn Blender in background mode twice per rig:
   a. Use the ActorsHQ `export_blender.py` helper to bake a .blend file that
      embeds the animated meshes together with the generated cameras.
   b. Render the requested frame range for every camera to RGB (optionally mask,
      depth) directories following the 4DGaussians multipleview layout.
4. Assemble dataset artefacts so 4DGaussians can consume them directly.
   - Copy rendered frames to `data/multipleview/<dataset>/<camXX>/frame_XXXXX.png`
   - Persist calibration (CSV + JSON) for downstream tooling.
   - Provide hooks to optionally trigger COLMAP / LLFF preprocessing once
     rendering is completed (mirrors `multipleviewprogress.sh`).

The script is designed to run on the workstation that already hosts:
- ActorsHQ sources at `~/repos/humanrf`
- gs7 camera simulation at `~/multiplexed-pixels/gs7`
- 4DGaussians checkout (this repository)
- Raw ActorsHQ sequences under `/share/monakhova/actorshq`

Typical usage (see `main` for CLI):
    python scripts/actorshq_resample.py \
        --actor Actor01 --sequence Sequence1 --scale 4x \
        --rig lightfield --rig stereo --rig iphone \
        --frames 0 10 \
        --output-root data/multipleview/actorshq_actor01_seq1

The implementation below progressively fills in each stage.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Utilities / argument parsing
# ---------------------------------------------------------------------------


def _extend_sys_path(actorshq_root: Path, gs7_root: Path) -> None:
    """Ensure humanrf/ActorsHQ and gs7 sources can be imported."""

    for candidate in (
        actorshq_root,
        actorshq_root / "actorshq",
        gs7_root,
    ):
        candidate = candidate.expanduser().resolve()
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


@dataclass(frozen=True)
class RigRequest:
    """Description of a virtual camera rig to synthesise."""

    kind: str  # lightfield, stereo, iphone

    def canonical_key(self) -> str:
        return "multiplexed" if self.kind == "lightfield" else self.kind


@dataclass
class ActorHqCamera:
    name: str
    width: int
    height: int
    rotation_axisangle: np.ndarray
    translation: np.ndarray
    focal_length: np.ndarray
    principal_point: np.ndarray

    @property
    def fx_pixel(self) -> float:
        return float(self.focal_length[0]) * float(self.width)

    @property
    def fy_pixel(self) -> float:
        return float(self.focal_length[1]) * float(self.height)

    @property
    def cx_pixel(self) -> float:
        return float(self.principal_point[0]) * float(self.width)

    @property
    def cy_pixel(self) -> float:
        return float(self.principal_point[1]) * float(self.height)

    def rotation_matrix_cam2world(self) -> np.ndarray:
        return axis_angle_to_matrix(self.rotation_axisangle)

    def extrinsic_matrix_cam2world(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = self.rotation_matrix_cam2world()
        matrix[:3, 3] = self.translation
        return matrix


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(rotvec, dtype=np.float64) / theta
    K = skew(axis)
    I = np.eye(3, dtype=np.float64)
    mat = I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return mat.astype(np.float64)


def matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    cos_theta = (np.trace(matrix) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)

    if abs(theta - math.pi) < 1e-5:
        # Handle angle close to pi separately for numerical stability.
        diag = np.clip((np.diagonal(matrix) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(diag)
        # Choose signs using off-diagonal elements.
        if matrix[0, 1] < 0:
            axis[1] = -axis[1]
        if matrix[0, 2] < 0:
            axis[2] = -axis[2]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            axis = np.array(
                [
                    math.sqrt(max(matrix[0, 0], 0.0)),
                    math.sqrt(max(matrix[1, 1], 0.0)),
                    math.sqrt(max(matrix[2, 2], 0.0)),
                ]
            )
            axis_norm = np.linalg.norm(axis)
        axis = axis / max(axis_norm, 1e-8)
    else:
        axis = np.array(
            [
                matrix[2, 1] - matrix[1, 2],
                matrix[0, 2] - matrix[2, 0],
                matrix[1, 0] - matrix[0, 1],
            ]
        )
        axis /= 2.0 * math.sin(theta)

    return (axis * theta).astype(np.float64)


def read_calibration_csv(csv_path: Path) -> List[ActorHqCamera]:
    cameras: List[ActorHqCamera] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            cameras.append(
                ActorHqCamera(
                    name=row["name"],
                    width=int(row["w"]),
                    height=int(row["h"]),
                    rotation_axisangle=np.array(
                        [float(row["rx"]), float(row["ry"]), float(row["rz"])],
                        dtype=np.float64,
                    ),
                    translation=np.array(
                        [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                        dtype=np.float64,
                    ),
                    focal_length=np.array(
                        [float(row["fx"]), float(row["fy"])], dtype=np.float64
                    ),
                    principal_point=np.array(
                        [float(row["px"]), float(row["py"])], dtype=np.float64
                    ),
                )
            )
    return cameras


def write_calibration_csv(cameras: Sequence[ActorHqCamera], csv_path: Path) -> None:
    csv_field_names = [
        "name",
        "w",
        "h",
        "rx",
        "ry",
        "rz",
        "tx",
        "ty",
        "tz",
        "fx",
        "fy",
        "px",
        "py",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        writer.writeheader()

        for camera in cameras:
            row = {
                "name": camera.name,
                "w": int(camera.width),
                "h": int(camera.height),
                "rx": float(camera.rotation_axisangle[0]),
                "ry": float(camera.rotation_axisangle[1]),
                "rz": float(camera.rotation_axisangle[2]),
                "tx": float(camera.translation[0]),
                "ty": float(camera.translation[1]),
                "tz": float(camera.translation[2]),
                "fx": float(camera.focal_length[0]),
                "fy": float(camera.focal_length[1]),
                "px": float(camera.principal_point[0]),
                "py": float(camera.principal_point[1]),
            }
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/share/monakhova/actorshq", type=Path)
    parser.add_argument("--actorshq-src", default="~/repos/humanrf", type=Path)
    parser.add_argument("--gs7-src", default="~/multiplexed-pixels/gs7", type=Path)

    parser.add_argument("--actor", required=True, help="Actor ID e.g. Actor01")
    parser.add_argument("--sequence", required=True, help="Sequence ID e.g. Sequence1")
    parser.add_argument(
        "--scale",
        default="4x",
        help="Spatial scale folder inside the sequence (1x,2x,4x).",
    )

    parser.add_argument(
        "--rig",
        action="append",
        choices=("lightfield", "multiplexed", "stereo", "iphone"),
        required=True,
        help="Rig generator to apply (repeat to request multiple rigs).",
    )
    parser.add_argument(
        "--multiplexed-count",
        type=int,
        default=16,
        help="Number of images for lightfield/multiplexed rigs (forms sqrt grid).",
    )
    parser.add_argument(
        "--lightfield-angle-deg",
        type=float,
        default=10.0,
        help="Coverage angle for lightfield cameras (deg).",
    )
    parser.add_argument(
        "--stereo-angle-deg",
        type=float,
        default=6.0,
        help="Coverage angle to space stereo views (deg).",
    )
    parser.add_argument(
        "--iphone-angle-deg",
        type=float,
        default=None,
        help="If provided, place iPhone trio using angular offsets instead of metric baselines.",
    )
    parser.add_argument(
        "--iphone-same-focal-lengths",
        action="store_true",
        help="Keep all iPhone cameras at reference focal length instead of 13/24/77mm equivalence.",
    )
    parser.add_argument(
        "--iphone-baseline-x",
        type=float,
        default=9.5,
        help="Metric baseline (mm) used when iphone-angle-deg is omitted.",
    )
    parser.add_argument(
        "--iphone-baseline-y",
        type=float,
        default=9.5,
        help="Metric baseline (mm) used when iphone-angle-deg is omitted.",
    )

    parser.add_argument(
        "--frames",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help="Optional frame window (inclusive start, exclusive end) to process.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Destination directory for generated dataset (inside 4DGS data/multipleview).",
    )
    parser.add_argument(
        "--blender-bin",
        default="blender",
        help="Path to Blender executable used for exports and renders.",
    )
    parser.add_argument(
        "--abc-object-path",
        default="/object",
        help="Object path inside meshes.abc when importing via Blender.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only build camera calibrations; do not launch Blender renders.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Render every Nth frame (default: 1, i.e., all frames).",
    )
    parser.add_argument(
        "--image-format",
        default="PNG",
        help="Image format for Blender renders (e.g., PNG, JPEG).",
    )
    parser.add_argument(
        "--color-mode",
        default="RGB",
        help="Blender color mode (RGB or RGBA).",
    )
    parser.add_argument(
        "--timeline-offset",
        type=int,
        default=1,
        help="Offset added to dataset frame index before addressing Blender timeline.",
    )
    parser.add_argument(
        "--filename-offset",
        type=int,
        default=1,
        help="Offset applied when numbering rendered frames on disk.",
    )
    parser.add_argument(
        "--n_train_views",
        type=int,
        default=-1,
        help="Number of base cameras to seed the rigs with (matches gs7 n_train_images logic).",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Core helpers (stage 2: camera rig generation)
# ---------------------------------------------------------------------------


def build_scene_info_from_actorhq(
    cameras: Sequence[ActorHqCamera],
    scale_root: Path,
) -> "SceneInfo":
    """Convert ActorsHQ calibration to gs7 SceneInfo (single-view per index)."""

    from scene.scene_utils import (
        BasicPointCloud,
        CameraInfo,
        SceneInfo,
        getNerfppNorm,
    )
    from utils.graphics_utils import focal2fov

    cam_infos: MutableMapping[int, List[CameraInfo]] = {}
    placeholder_image = Image.new(
        "RGB", (cameras[0].width, cameras[0].height), color=(0, 0, 0)
    )

    for idx, cam in enumerate(cameras):
        c2w = cam.extrinsic_matrix_cam2world()
        w2c = np.linalg.inv(c2w)
        R_stored = w2c[:3, :3].T
        T_stored = w2c[:3, 3]

        fov_x = focal2fov(cam.fx_pixel, cam.width)
        fov_y = focal2fov(cam.fy_pixel, cam.height)

        cam_info = CameraInfo(
            uid=idx,
            groupid=idx,
            R=R_stored,
            T=T_stored,
            FovY=fov_y,
            FovX=fov_x,
            image=placeholder_image,
            image_path=str(
                scale_root / "rgbs" / cam.name / f"{cam.name}_rgb000000.jpg"
            ),
            image_name=cam.name,
            width=cam.width,
            height=cam.height,
            mask=None,
            mask_name="",
        )
        cam_infos[idx] = [cam_info]

    nerf_norm = getNerfppNorm([cam for cams in cam_infos.values() for cam in cams])
    empty_cloud = BasicPointCloud(
        points=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
    )
    return SceneInfo(
        point_cloud=empty_cloud,
        train_cameras=cam_infos,
        test_cameras=[],
        full_test_cameras=[],
        nerf_normalization=nerf_norm,
        ply_path=str(scale_root / "points3d_placeholder.ply"),
    )


def estimate_object_center(aabb_csv: Path) -> np.ndarray:
    """Average centre of bounding boxes (actorshq/dataset/aabb_data)."""

    from actorshq.dataset.aabb_data import read_aabbs_csv

    aabbs = read_aabbs_csv(aabb_csv)
    if not aabbs:
        raise RuntimeError(f"No aabb entries found in {aabb_csv}")
    centers = [0.5 * (entry.aabb[0] + entry.aabb[1]) for entry in aabbs]
    return np.mean(np.stack(centers, axis=0), axis=0)


def iter_scene_cameras(scene_info: "SceneInfo"):
    """Yield (view_idx, per_view_index, CameraInfo)."""

    for view_idx in sorted(scene_info.train_cameras.keys()):
        cam_list = scene_info.train_cameras[view_idx]
        for cam_idx, cam_info in enumerate(cam_list):
            yield view_idx, cam_idx, cam_info


def camera_info_to_camera_data(
    cam_info: "CameraInfo",
    name: str,
) -> ActorHqCamera:
    """Convert gs7 CameraInfo to ActorsHQ-style calibration entry."""

    from scene.scene_utils import camera_center_world
    from utils.graphics_utils import fov2focal

    rotation_axisangle = matrix_to_axis_angle(np.asarray(cam_info.R))
    translation = camera_center_world(cam_info)

    fx_px = fov2focal(cam_info.FovX, cam_info.width)
    fy_px = fov2focal(cam_info.FovY, cam_info.height)
    focal = np.array(
        [fx_px / float(cam_info.width), fy_px / float(cam_info.height)], dtype=np.float64
    )
    principal = np.array([0.5, 0.5], dtype=np.float64)

    return ActorHqCamera(
        name=name,
        width=int(cam_info.width),
        height=int(cam_info.height),
        rotation_axisangle=rotation_axisangle.astype(np.float64),
        translation=translation.astype(np.float64),
        focal_length=focal,
        principal_point=principal,
    )


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def write_transforms_json(
    rig_dir: Path,
    actor_cameras: Dict[str, ActorHqCamera],
    frame_start: int,
    frame_end: int,
    frame_step: int,
    filename_offset: int,
) -> Path:
    if not actor_cameras:
        raise ValueError("No cameras provided to generate transforms.json")

    sorted_names = sorted(actor_cameras.keys())
    first_cam = actor_cameras[sorted_names[0]]

    fl_x0 = first_cam.fx_pixel
    fl_y0 = first_cam.fy_pixel
    width0 = first_cam.width
    height0 = first_cam.height

    transforms_payload: Dict[str, object] = {
        "camera_angle_x": 2.0 * math.atan(width0 / (2.0 * fl_x0)),
        "camera_angle_y": 2.0 * math.atan(height0 / (2.0 * fl_y0)),
        "fl_x": fl_x0,
        "fl_y": fl_y0,
        "cx": first_cam.cx_pixel,
        "cy": first_cam.cy_pixel,
        "w": width0,
        "h": height0,
        "frames": [],
    }

    for cam_name in sorted_names:
        actor_cam = actor_cameras[cam_name]
        c2w = actor_cam.extrinsic_matrix_cam2world().astype(np.float64)

        for dataset_frame in range(frame_start, frame_end, frame_step):
            file_index = dataset_frame + filename_offset
            frame_entry = {
                "file_path": f"{cam_name}/frame_{file_index:05d}.png",
                "transform_matrix": c2w.tolist(),
                "fl_x": actor_cam.fx_pixel,
                "fl_y": actor_cam.fy_pixel,
                "cx": actor_cam.cx_pixel,
                "cy": actor_cam.cy_pixel,
                "w": actor_cam.width,
                "h": actor_cam.height,
                "split": "train",
            }
            transforms_payload["frames"].append(frame_entry)

    transforms_path = rig_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms_payload, indent=2), encoding="utf-8")
    return transforms_path


def restrict_train_views(
    scene_info: "SceneInfo", n_train_views: int
) -> Tuple["SceneInfo", List[int]]:
    """Select a subset of main training views using max-min dispersion."""

    if n_train_views is None or n_train_views < 1:
        selected = sorted(scene_info.train_cameras.keys())
        return scene_info, selected

    from scene.scene_utils import camera_center_world
    from utils.render_utils import find_max_min_dispersion_subset

    sorted_items = sorted(scene_info.train_cameras.items(), key=lambda item: item[0])
    cam_entries: List[Tuple[int, "CameraInfo"]] = []
    for view_idx, cam_list in sorted_items:
        if not cam_list:
            continue
        cam_entries.append((view_idx, cam_list[0]))

    if n_train_views >= len(cam_entries):
        selected = [view_idx for view_idx, _ in cam_entries]
        return scene_info, selected

    centers = np.stack([camera_center_world(cam) for _, cam in cam_entries], axis=0)
    anchor_uid = cam_entries[0][1].uid
    try:
        anchor_index = [cam.uid for _, cam in cam_entries].index(anchor_uid)
    except ValueError:
        anchor_index = None

    chosen_indices = find_max_min_dispersion_subset(
        centers, int(n_train_views), anchor_index
    )
    selected_view_indices = [cam_entries[i][0] for i in chosen_indices]
    new_train = {
        view_idx: scene_info.train_cameras[view_idx]
        for view_idx in selected_view_indices
    }
    return scene_info._replace(train_cameras=new_train), selected_view_indices


BLENDER_RENDER_TEMPLATE = textwrap.dedent(
    """
    import argparse
    import json
    import sys
    from pathlib import Path

    import bpy


    def ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)


    def pick_scene(camera_obj):
        if getattr(camera_obj, "users_scene", None):
            scenes = camera_obj.users_scene
            if scenes:
                return scenes[0]
        return bpy.context.scene


    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-end", type=int, required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--timeline-offset", type=int, default=1)
    parser.add_argument("--filename-offset", type=int, default=1)
    parser.add_argument("--image-format", default="PNG")
    parser.add_argument("--color-mode", default="RGB")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    args = parser.parse_args(argv)

    with open(args.metadata, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cameras = payload.get("cameras", [])

    frames = list(range(args.frame_start, args.frame_end, args.frame_step))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for scene in bpy.data.scenes:
        scene.render.use_multiview = False
        scene.render.image_settings.file_format = args.image_format
        scene.render.image_settings.color_mode = args.color_mode

    for entry in cameras:
        camera_name = entry["name"]
        cam_obj = bpy.data.objects.get(f"cam_{camera_name}")
        if cam_obj is None:
            raise RuntimeError(f\"Camera object cam_{camera_name} not found in blend file.\")
        scene = pick_scene(cam_obj)
        scene.camera = cam_obj
        camera_dir = output_root / camera_name
        ensure_dir(camera_dir)

        for dataset_frame in frames:
            timeline_frame = dataset_frame + args.timeline_offset
            filename_index = dataset_frame + args.filename_offset
            scene.frame_set(timeline_frame)
            scene.render.filepath = str(camera_dir / f"frame_{filename_index:05d}")
            bpy.ops.render.render(write_still=True, scene=scene.name)
    """
)


def run_cmd(command: Sequence[str], cwd: Optional[Path] = None) -> None:
    command_str = " ".join(str(part) for part in command)
    print(f"[cmd] {command_str}")
    subprocess.run(command, cwd=cwd, check=True)


def write_temp_render_script() -> Path:
    with tempfile.NamedTemporaryFile(
        "w", suffix="_actorshq_render.py", delete=False
    ) as handle:
        handle.write(BLENDER_RENDER_TEMPLATE)
        temp_path = Path(handle.name)
    return temp_path


def create_blend_scene(
    blender_bin: str,
    export_script: Path,
    calibration_csv: Path,
    abc_path: Path,
    blend_path: Path,
    abc_object_path: str,
) -> None:
    command = [
        blender_bin,
        "--background",
        "--python",
        str(export_script),
        "--",
        "--csv",
        str(calibration_csv),
        "--abc",
        str(abc_path),
        "--abc_object_path",
        abc_object_path,
        "--blend",
        str(blend_path),
    ]
    run_cmd(command, cwd=export_script.parent)


def render_rig_with_blender(
    blender_bin: str,
    render_script: Path,
    output_root: Path,
    frame_start: int,
    frame_end: int,
    frame_step: int,
    timeline_offset: int,
    filename_offset: int,
    image_format: str,
    color_mode: str,
    rig_name: str,
    camera_entries: List[dict],
    blend_template: Path,
) -> None:
    """Render cameras for a rig using a shared Blender scene file."""

    metadata_json = output_root / "camera_metadata.json"
    metadata_json.write_text(json.dumps({"cameras": camera_entries}, indent=2), encoding="utf-8")

    print(
        f"[rig:{rig_name}] Rendering frames {frame_start}..{frame_end - 1} step {frame_step}"
    )

    command = [
        blender_bin,
        "-b",
        str(blend_template),
        "--python",
        str(render_script),
        "--",
        "--metadata",
        str(metadata_json),
        "--output-root",
        str(output_root),
        "--frame-start",
        str(frame_start),
        "--frame-end",
        str(frame_end),
        "--frame-step",
        str(frame_step),
        "--timeline-offset",
        str(timeline_offset),
        "--filename-offset",
        str(filename_offset),
        "--image-format",
        image_format,
        "--color-mode",
        color_mode,
    ]
    run_cmd(command, cwd=blend_template.parent)


def determine_frame_range(
    frame_arg: Optional[Sequence[int]], scene_json: Path
) -> Tuple[int, int]:
    if frame_arg is not None:
        start, end = map(int, frame_arg)
    else:
        with open(scene_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        num_frames = int(payload.get("num_frames", 0))
        start, end = 0, num_frames

    if end <= start:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")
    return start, end


# ---------------------------------------------------------------------------
# CLI entry point + rig generation orchestration
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    dataset_root: Path = Path(args.dataset_root).expanduser().resolve()
    actorshq_src: Path = Path(args.actorshq_src).expanduser().resolve()
    gs7_src: Path = Path(args.gs7_src).expanduser().resolve()
    _extend_sys_path(actorshq_src, gs7_src)

    # Lazy imports after path manipulation
    from scene.dataset_readers import (
        create_iphone_views,
        create_multiplexed_views,
        create_stereo_views,
    )
    from scene.scene_utils import SceneInfo

    sequence_root = dataset_root / args.actor / args.sequence
    scale_root = sequence_root / args.scale
    calibration_csv = scale_root / "calibration.csv"
    aabb_csv = sequence_root / "aabbs.csv"
    meshes_abc = sequence_root / "meshes.abc"
    scene_json = sequence_root / "scene.json"
    export_script = actorshq_src / "actorshq" / "toolbox" / "export_blender.py"

    if not calibration_csv.exists():
        raise FileNotFoundError(f"Could not find calibration csv at {calibration_csv}")
    if not aabb_csv.exists():
        raise FileNotFoundError(f"Could not find aabbs csv at {aabb_csv}")
    if not meshes_abc.exists():
        raise FileNotFoundError(f"Could not find meshes.abc at {meshes_abc}")
    if not scene_json.exists():
        raise FileNotFoundError(f"Could not find scene.json at {scene_json}")
    if not export_script.exists():
        raise FileNotFoundError(
            f"export_blender.py missing at expected location {export_script}"
        )

    frame_start, frame_end = determine_frame_range(args.frames, scene_json)
    blender_bin = str(args.blender_bin)
    image_format = str(args.image_format).upper()
    color_mode = str(args.color_mode).upper()

    if args.frame_step <= 0:
        raise ValueError("frame-step must be >= 1")

    render_script_path: Optional[Path] = None
    try:
        if not args.skip_render:
            render_script_path = write_temp_render_script()

        cameras = read_calibration_csv(calibration_csv)
        base_scene = build_scene_info_from_actorhq(cameras, scale_root)
        base_scene, selected_base_views = restrict_train_views(
            base_scene, int(args.n_train_views)
        )

        object_center = estimate_object_center(aabb_csv)

        num_frames_total = (frame_end - frame_start + int(args.frame_step) - 1) // int(args.frame_step)
        print(f"Loaded {len(cameras)} cameras for base scene.")
        print(f"Estimated object centre (world coords): {object_center.tolist()}")
        print(
            f"Rendering coverage would include {num_frames_total} frames per camera "
            f"(range {frame_start}–{frame_end - 1} step {args.frame_step})."
        )
        if int(args.n_train_views) > 0:
            print(
                f"Selected {len(selected_base_views)} primary views for rig generation: {selected_base_views}"
            )

        rig_requests = [RigRequest(kind=r) for r in (args.rig or ())]

        rig_results: Dict[str, SceneInfo] = {}

        for request in rig_requests:
            if request.kind in rig_results:
                print(
                    f"[rig:{request.kind}] already generated, skipping duplicate entry."
                )
                continue

            canonical = request.canonical_key()
            if canonical == "multiplexed":
                angle = float(args.lightfield_angle_deg)
                count = int(args.multiplexed_count)
                scene_variant = create_multiplexed_views(
                    base_scene,
                    obj_center=object_center,
                    angle_deg=angle,
                    n_multiplexed_images=count,
                )
                print(
                    f"[rig:{request.kind}] angle={angle:.2f}°, cameras={sum(len(v) for v in scene_variant.train_cameras.values())}"
                )
            elif canonical == "stereo":
                angle = float(args.stereo_angle_deg)
                scene_variant = create_stereo_views(
                    base_scene, obj_center=object_center, angle_deg=angle
                )
                print(
                    f"[rig:{request.kind}] angle={angle:.2f}°, #views={len(scene_variant.train_cameras)}"
                )
            elif canonical == "iphone":
                angle = (
                    None if args.iphone_angle_deg is None else float(args.iphone_angle_deg)
                )
                scene_variant = create_iphone_views(
                    base_scene,
                    obj_center=object_center,
                    angle_deg=angle,
                    same_focal_lengths=args.iphone_same_focal_lengths,
                    baseline_x=float(args.iphone_baseline_x),
                    baseline_y=float(args.iphone_baseline_y),
                )
                print(f"[rig:{request.kind}] views={len(scene_variant.train_cameras)}")
            else:
                raise ValueError(f"Unsupported rig kind: {request.kind}")

            rig_results[request.kind] = scene_variant

        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        rig_calibration_summaries: Dict[str, Dict[str, object]] = {}
        rig_details: Dict[str, Dict[str, object]] = {}
        all_camera_datas: List[ActorHqCamera] = []

        for label, scene_info in rig_results.items():
            rig_out_dir = output_root / label
            rig_out_dir.mkdir(parents=True, exist_ok=True)

            camera_datas = []
            metadata_entries = []
            for view_idx, cam_idx, cam_info in iter_scene_cameras(scene_info):
                base_name = cam_info.image_name or f"view{view_idx:03d}"
                camera_name = sanitize_name(
                    f"{label}_v{view_idx:03d}_c{cam_idx:02d}_{base_name}"
                )
                camera_data = camera_info_to_camera_data(cam_info, camera_name)
                camera_datas.append(camera_data)
                metadata_entries.append(
                    {
                        "name": camera_name,
                        "view_index": int(view_idx),
                        "groupid": int(getattr(cam_info, "groupid", view_idx)),
                        "uid": int(getattr(cam_info, "uid", view_idx)),
                        "source_image_name": cam_info.image_name,
                        "fov_x_rad": float(cam_info.FovX),
                        "fov_y_rad": float(cam_info.FovY),
                        "width": int(cam_info.width),
                        "height": int(cam_info.height),
                    }
                )

            if not camera_datas:
                print(f"[rig:{label}] No cameras generated, skipping export.")
                continue

            calibration_path = rig_out_dir / "calibration.csv"
            write_calibration_csv(camera_datas, calibration_path)
            metadata_path = rig_out_dir / "camera_metadata.json"
            metadata_path.write_text(
                json.dumps({"cameras": metadata_entries}, indent=2), encoding="utf-8"
            )

            rig_calibration_summaries[label] = {
                "output_dir": str(rig_out_dir),
                "num_cameras": len(camera_datas),
                "calibration_csv": str(calibration_path),
                "metadata_json": str(metadata_path),
            }
            actor_map = {cam.name: cam for cam in camera_datas}
            rig_details[label] = {
                "scene_info": scene_info,
                "out_dir": rig_out_dir,
                "actor_cameras": actor_map,
                "metadata": metadata_entries,
            }
            write_transforms_json(
                rig_out_dir,
                actor_map,
                frame_start,
                frame_end,
                int(args.frame_step),
                int(args.filename_offset),
            )
            all_camera_datas.extend(camera_datas)

        shared_blend_path: Optional[Path] = None
        if all_camera_datas:
            shared_calibration_path = output_root / "calibration_all_rigs.csv"
            write_calibration_csv(all_camera_datas, shared_calibration_path)
            if not args.skip_render:
                shared_blend_path = output_root / f"{args.actor}_{args.sequence}.blend"
                print(
                    f"[shared] Exporting Blender scene with {len(all_camera_datas)} cameras to {shared_blend_path}"
                )
                create_blend_scene(
                    blender_bin,
                    export_script,
                    shared_calibration_path,
                    meshes_abc,
                    shared_blend_path,
                    args.abc_object_path,
                )

        if not args.skip_render and shared_blend_path is not None:
            print(
                f"[shared] Rendering frames {frame_start}..{frame_end - 1} step {args.frame_step} for each rig"
            )
            for label, details in rig_details.items():
                rig_out_dir = details["out_dir"]
                metadata_entries = details["metadata"]
                render_rig_with_blender(
                    blender_bin,
                    render_script_path,
                    rig_out_dir,
                    frame_start,
                    frame_end,
                    int(args.frame_step),
                    int(args.timeline_offset),
                    int(args.filename_offset),
                    image_format,
                    color_mode,
                    label,
                    metadata_entries,
                    shared_blend_path,
                )
                rig_calibration_summaries[label].update(
                    {
                        "blend_file": str(shared_blend_path),
                        "images_root": str(rig_out_dir),
                        "frame_range": [frame_start, frame_end],
                        "frame_step": int(args.frame_step),
                    }
                )

        output_summary = {}
        for label, scene_info in rig_results.items():
            base_info = {
                "num_views": sum(len(v) for v in scene_info.train_cameras.values()),
                "view_indices": sorted(scene_info.train_cameras.keys()),
            }
            base_info.update(rig_calibration_summaries.get(label, {}))
            output_summary[label] = base_info

        print(json.dumps(output_summary, indent=2))
    finally:
        if render_script_path and render_script_path.exists():
            render_script_path.unlink()


if __name__ == "__main__":
    main()
