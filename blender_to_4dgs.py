#!/usr/bin/env python3
"""Convert Blender multi-camera outputs to the 4DGaussians multi-view format.

This variant relies on VGGT (demo_colmap.py) to generate an aligned point cloud
while using the ground-truth Blender camera poses for all COLMAP artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def ensure_cudss_accessible() -> None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return

    cudss_dirs = sorted(
        Path(prefix).glob("opt/libcudss-linux-x86_64-*_cuda12-archive")
    )
    if not cudss_dirs:
        return

    cudss_dir = cudss_dirs[-1]
    env = os.environ
    env.setdefault("CUDSS_DIR", str(cudss_dir))
    env.setdefault("cudss_DIR", str(cudss_dir / "lib/cmake/cudss"))

    prefix_path = env.get("CMAKE_PREFIX_PATH")
    if prefix_path:
        parts = prefix_path.split(":")
        if str(cudss_dir) not in parts:
            env["CMAKE_PREFIX_PATH"] = ":".join([str(cudss_dir), prefix_path])
    else:
        env["CMAKE_PREFIX_PATH"] = str(cudss_dir)

    cudss_lib = str(cudss_dir / "lib")
    ld_library_path = env.get("LD_LIBRARY_PATH")
    if ld_library_path:
        parts = ld_library_path.split(":")
        if cudss_lib not in parts:
            env["LD_LIBRARY_PATH"] = ":".join([cudss_lib, ld_library_path])
    else:
        env["LD_LIBRARY_PATH"] = cudss_lib

    libcudss = cudss_dir / "lib" / "libcudss.so.0"
    if libcudss.exists():
        try:
            ctypes.CDLL(str(libcudss))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to preload cuDSS library at {libcudss}: {exc}"
            ) from exc


ensure_cudss_accessible()

try:  # Users may skip VGGT / reconstruction in dry runs
    import pycolmap  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    pycolmap = None  # type: ignore

from PIL import Image
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary  # noqa: E402

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")


@dataclass
class FrameRecord:
    camera_name: str
    camera_original_name: str
    frame_index: int
    dst_path: Path
    transform_matrix: np.ndarray
    width: int
    height: int
    tmp_basename: str | None = None


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def extract_camera_name(file_path: str) -> str:
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return parts[-2]
    return "camera_00"


def parse_frame_index(file_path: str) -> int:
    tokens = Path(file_path).stem.split("_")
    for token in reversed(tokens):
        if token.isdigit():
            return int(token)
        if token.startswith("r") and token[1:].isdigit():
            return int(token[1:])
    digits = [c for c in Path(file_path).stem if c.isdigit()]
    return int("".join(digits)) if digits else 0


def resolve_source_image(blender_dir: Path, file_path: str) -> Path:
    rel = Path(file_path)
    if rel.is_absolute():
        candidate = rel
    else:
        parts = [part for part in rel.parts if part not in (".", "")]
        candidate = blender_dir.joinpath(*parts)

    if candidate.exists():
        return candidate

    core = candidate.with_suffix("")
    for ext in SUPPORTED_IMAGE_EXTS:
        probe = core.with_suffix(ext)
        if probe.exists():
            return probe

    matches = sorted(core.parent.glob(core.name + ".*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not locate source image for {file_path}")


def convert_image_to_jpeg(src: Path, dst: Path) -> Tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        width, height = img.size
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst, format="JPEG", quality=95)
    return width, height


def copy_frames_to_dataset(
    blender_dir: Path, output_dir: Path, transforms: Dict
) -> Dict[str, Dict]:
    frames = transforms.get("frames", [])
    if not frames:
        raise ValueError("transforms.json does not contain any frames")

    grouped: Dict[str, List[Dict]] = {}
    for frame in frames:
        cam_name = extract_camera_name(frame["file_path"])
        grouped.setdefault(cam_name, []).append(frame)

    metadata: Dict[str, Dict] = {}
    for cam_idx, cam_name in enumerate(sorted(grouped.keys())):
        canonical_cam = f"cam_{cam_idx + 1:05d}"
        cam_dir = output_dir / canonical_cam
        cam_dir.mkdir(parents=True, exist_ok=True)

        ordered_frames = sorted(
            grouped[cam_name], key=lambda f: parse_frame_index(f["file_path"])
        )
        frame_records: List[FrameRecord] = []

        for seq_idx, frame in enumerate(ordered_frames):
            src_path = resolve_source_image(blender_dir, frame["file_path"])
            dst_path = cam_dir / f"frame_{seq_idx + 1:05d}.jpg"
            width, height = convert_image_to_jpeg(src_path, dst_path)
            frame_records.append(
                FrameRecord(
                    camera_name=canonical_cam,
                    camera_original_name=cam_name,
                    frame_index=parse_frame_index(frame["file_path"]),
                    dst_path=dst_path,
                    transform_matrix=np.array(
                        frame["transform_matrix"], dtype=np.float64
                    ),
                    width=width,
                    height=height,
                )
            )

        if not frame_records:
            raise RuntimeError(f"No frames copied for camera {cam_name}")

        metadata[canonical_cam] = {
            "original_name": cam_name,
            "frames": frame_records,
        }

    return metadata


def gather_all_frames(metadata: Dict[str, Dict]) -> List[FrameRecord]:
    frames: List[FrameRecord] = []
    for cam_name in sorted(metadata.keys()):
        frames.extend(metadata[cam_name]["frames"])
    frames.sort(key=lambda fr: (fr.frame_index, fr.camera_name))
    return frames


def select_first_frame_per_camera(metadata: Dict[str, Dict]) -> List[FrameRecord]:
    selected: List[FrameRecord] = []
    for cam_name in sorted(metadata.keys()):
        frames = metadata[cam_name]["frames"]
        frame = min(frames, key=lambda fr: fr.frame_index)
        selected.append(frame)
    return selected


def prepare_vggt_scene(frames: Iterable[FrameRecord], scene_dir: Path) -> None:
    images_dir = scene_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        basename = f"{frame.camera_name}_{frame.dst_path.name}"
        shutil.copy2(frame.dst_path, images_dir / basename)
        frame.tmp_basename = basename


def run_vggt_demo(demo_script: Path, scene_dir: Path, conda_env: Optional[str]) -> None:
    if not demo_script.exists():
        raise FileNotFoundError(f"VGGT script not found at {demo_script}")

    if conda_env:
        cmd = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            str(demo_script),
            "--scene_dir",
            str(scene_dir),
            "--stage",
            "vggt",
        ]
    else:
        cmd = [
            sys.executable,
            str(demo_script),
            "--scene_dir",
            str(scene_dir),
            "--stage",
            "vggt",
        ]
    print("->", " ".join(cmd))
    result = subprocess.run(cmd, cwd=demo_script.parent, check=False)
    if result.returncode != 0:
        raise RuntimeError("VGGT demo_colmap execution failed")


def load_vggt_outputs(
    scene_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    sparse_dir = scene_dir / "sparse"
    points_path = sparse_dir / "points.ply"
    if not points_path.exists():
        raise FileNotFoundError(f"VGGT did not produce a point cloud at {points_path}")

    ply = PlyData.read(str(points_path))
    vertices = ply["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(
        np.float64
    )
    colors = np.stack(
        [vertices["red"], vertices["green"], vertices["blue"]], axis=1
    ).astype(np.uint8)

    cam_extrinsics = read_extrinsics_binary(sparse_dir / "images.bin")
    pose_by_name: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for image in cam_extrinsics.values():
        R_wc = qvec2rotmat(image.qvec)
        t_wc = image.tvec
        pose_by_name[Path(image.name).name] = (R_wc, t_wc)

    return points, colors, pose_by_name


def camera_center_from_colmap(R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    return (-R_wc.T @ t_wc.reshape(3, 1)).reshape(3)


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    assert src.shape == dst.shape
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = src_centered.T @ dst_centered / src.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_src = np.sum(src_centered**2) / src.shape[0]
    scale = np.sum(S) / var_src if var_src > 0 else 1.0
    t = mean_dst - scale * R @ mean_src
    return scale, R, t


def align_point_cloud(
    points: np.ndarray,
    vggt_centers: np.ndarray,
    blender_centers: np.ndarray,
) -> np.ndarray:
    scale, R, t = umeyama_alignment(vggt_centers, blender_centers)
    aligned = (scale * (R @ points.T)).T + t
    return aligned


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    assert points.shape[0] == colors.shape[0]
    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["nx"] = 0
    vertex["ny"] = 0
    vertex["nz"] = 0
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    ply = PlyData([PlyElement.describe(vertex, "vertex")], text=True)
    ply.write(str(path))


def maybe_downsample(
    points: np.ndarray, colors: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        voxel = 0.01
        while np.asarray(pcd.points).shape[0] > max_points:
            pcd = pcd.voxel_down_sample(voxel)
            voxel *= 1.2
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
        return points, colors
    except ImportError:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        return points[idx], colors[idx]


def blender_matrix_to_opencv_c2w(matrix: np.ndarray) -> np.ndarray:
    c2w = matrix.copy()
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    return c2w


def create_reconstruction(
    frames: List[FrameRecord],
    transforms: Dict,
    sparse_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    use_pycolmap = False
    cam_attr = None
    if pycolmap is not None:
        cam_attr = getattr(pycolmap.Image, "cam_from_world", None)
        setter = getattr(cam_attr, "fset", None) if cam_attr is not None else None
        use_pycolmap = setter is not None

    if use_pycolmap:
        reconstruction = pycolmap.Reconstruction()

        cameras: Dict[str, pycolmap.Camera] = {}
        for frame in frames:
            if frame.camera_name in cameras:
                continue
            width, height = frame.width, frame.height
            if "fl_x" in transforms and "fl_y" in transforms:
                fx = float(transforms["fl_x"])
                fy = float(transforms["fl_y"])
                cx = float(transforms.get("cx", width / 2))
                cy = float(transforms.get("cy", height / 2))
                camera = pycolmap.Camera(
                    model="PINHOLE",
                    width=width,
                    height=height,
                    params=[fx, fy, cx, cy],
                    camera_id=len(cameras) + 1,
                )
            else:
                if "camera_angle_x" not in transforms:
                    raise ValueError(
                        "Unable to determine focal length from transforms.json"
                    )
                fx = width / (2 * math.tan(float(transforms["camera_angle_x"]) / 2))
                cx = width / 2
                cy = height / 2
                camera = pycolmap.Camera(
                    model="SIMPLE_PINHOLE",
                    width=width,
                    height=height,
                    params=[fx, cx, cy],
                    camera_id=len(cameras) + 1,
                )
            reconstruction.add_camera(camera)
            cameras[frame.camera_name] = camera

        for image_id, frame in enumerate(frames, start=1):
            c2w = blender_matrix_to_opencv_c2w(frame.transform_matrix)
            w2c = np.linalg.inv(c2w)
            R_wc = w2c[:3, :3]
            t_wc = w2c[:3, 3]
            image = pycolmap.Image()
            image.image_id = image_id
            image.name = str(Path(frame.camera_name) / frame.dst_path.name)
            image.camera_id = cameras[frame.camera_name].camera_id
            cam_from_world = pycolmap.Rigid3d(np.hstack([R_wc, t_wc.reshape(3, 1)]))
            image.cam_from_world = cam_from_world
            image.registered = True
            reconstruction.add_image(image)

        if points.size > 0:
            sample_count = min(len(points), 5000)
            sample_idx = np.linspace(0, len(points) - 1, sample_count).astype(np.int64)
            for idx in sample_idx:
                reconstruction.add_point3D(points[idx], pycolmap.Track(), colors[idx])

        reconstruction.write(sparse_dir)
        return

    try:
        from colmap_converter import (
            Camera as ColmapCamera,
            Image as ColmapImage,
            Point3D as ColmapPoint3D,
            write_cameras_binary,
            write_cameras_text,
            write_images_binary,
            write_images_text,
            write_points3D_binary,
            write_points3D_text,
        )
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "colmap_converter.py is required to export COLMAP artifacts when pycolmap "
            "does not expose pose setters"
        ) from exc

    camera_id_map: Dict[str, int] = {}
    cameras_dict: Dict[int, ColmapCamera] = {}
    for frame in frames:
        if frame.camera_name in camera_id_map:
            continue
        width, height = frame.width, frame.height
        if "fl_x" in transforms and "fl_y" in transforms:
            fx = float(transforms["fl_x"])
            fy = float(transforms["fl_y"])
            cx = float(transforms.get("cx", width / 2))
            cy = float(transforms.get("cy", height / 2))
            model = "PINHOLE"
            params = np.array([fx, fy, cx, cy], dtype=np.float64)
        else:
            if "camera_angle_x" not in transforms:
                raise ValueError("Unable to determine focal length from transforms.json")
            fx = width / (2 * math.tan(float(transforms["camera_angle_x"]) / 2))
            cx = width / 2
            cy = height / 2
            model = "SIMPLE_PINHOLE"
            params = np.array([fx, cx, cy], dtype=np.float64)

        camera_id = len(camera_id_map) + 1
        camera_id_map[frame.camera_name] = camera_id
        cameras_dict[camera_id] = ColmapCamera(
            id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )

    images_dict: Dict[int, ColmapImage] = {}
    for image_id, frame in enumerate(frames, start=1):
        c2w = blender_matrix_to_opencv_c2w(frame.transform_matrix)
        w2c = np.linalg.inv(c2w)
        R_wc = w2c[:3, :3]
        t_wc = w2c[:3, 3]
        qvec = rotmat2qvec(R_wc).astype(np.float64)
        tvec = t_wc.astype(np.float64)
        images_dict[image_id] = ColmapImage(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id_map[frame.camera_name],
            name=str(Path(frame.camera_name) / frame.dst_path.name),
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros(0, dtype=np.int64),
        )

    points_dict: Dict[int, ColmapPoint3D] = {}
    if points.size > 0:
        sample_count = min(len(points), 5000)
        sample_idx = np.linspace(0, len(points) - 1, sample_count).astype(np.int64)
        for point_id, idx in enumerate(sample_idx, start=1):
            xyz = points[idx].astype(np.float64)
            rgb = colors[idx].astype(np.uint8)
            points_dict[point_id] = ColmapPoint3D(
                id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=0.0,
                image_ids=np.zeros(0, dtype=np.int32),
                point2D_idxs=np.zeros(0, dtype=np.int32),
            )

    write_cameras_binary(cameras_dict, str(sparse_dir / "cameras.bin"))
    write_images_binary(images_dict, str(sparse_dir / "images.bin"))
    write_points3D_binary(points_dict, str(sparse_dir / "points3D.bin"))
    write_cameras_text(cameras_dict, str(sparse_dir / "cameras.txt"))
    write_images_text(images_dict, str(sparse_dir / "images.txt"))
    write_points3D_text(points_dict, str(sparse_dir / "points3D.txt"))


def compute_poses_bounds_from_frames(
    frames: List[FrameRecord],
    points: np.ndarray,
    transforms: Dict,
    output_path: Path,
) -> None:
    data = []
    point_cloud = points.T
    for frame in frames:
        c2w = blender_matrix_to_opencv_c2w(frame.transform_matrix)
        w2c = np.linalg.inv(c2w)
        R_wc = w2c[:3, :3]
        t_wc = w2c[:3, 3]
        pose = np.zeros((3, 5), dtype=np.float32)
        pose[:, :3] = c2w[:3, :3]
        pose[:, 3] = c2w[:3, 3]

        if "fl_x" in transforms and "fl_y" in transforms:
            fx = float(transforms["fl_x"])
            fy = float(transforms["fl_y"])
        else:
            fx = frame.width / (2 * math.tan(float(transforms["camera_angle_x"]) / 2))
            fy = fx

        pose[:, 4] = np.array([frame.height, frame.width, fx], dtype=np.float32)

        depths = (R_wc @ point_cloud + t_wc.reshape(3, 1))[2]
        positive = depths[depths > 0]
        if positive.size == 0:
            near, far = 0.1, 10.0
        else:
            near = float(np.percentile(positive, 1) * 0.9)
            far = float(np.percentile(positive, 99) * 1.1)

        data.append(
            np.concatenate([pose.reshape(-1), np.array([near, far], dtype=np.float32)])
        )

    np.save(output_path, np.stack(data))


def create_config_file(output_dir: Path, dataset_name: str) -> None:
    config_dir = REPO_ROOT / "arguments" / "multipleview"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}.py"

    content = """ModelHiddenParams = dict(
    kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 150]
    },
    multires = [1, 2],
    defor_depth = 0,
    net_width = 128,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,
    no_do = False,
    no_dshs = False,
    no_ds = False,
    empty_voxel = False,
    render_process = False,
    static_mlp = False,
)

OptimizationParams = dict(
    dataloader = True,
    iterations = 15000,
    batch_size = 1,
    coarse_iterations = 3000,
    densify_until_iter = 10000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
)
"""

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Blender multi-cam renders to 4DGaussians multi-view format",
    )
    parser.add_argument(
        "--blender_dir", required=True, help="Directory containing Blender outputs"
    )
    parser.add_argument("--output_dir", required=True, help="Target dataset directory")
    parser.add_argument(
        "--dataset_name", help="Optional dataset name for config generation"
    )
    parser.add_argument(
        "--vggt_script",
        type=Path,
        default=None,
        help="Path to VGGT demo_colmap.py (required unless --skip_vggt is set)",
    )
    parser.add_argument(
        "--skip_vggt",
        action="store_true",
        help="Assume points3D_multipleview.ply already exists and skip VGGT",
    )
    parser.add_argument(
        "--max_point_cloud_points",
        type=int,
        default=40000,
        help="Downsample target for the fused point cloud (0 disables)",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep the temporary tmp_colmap directory instead of deleting it",
    )
    parser.add_argument(
        "--vggt_conda_env",
        type=str,
        default="transformers",
        help="Conda environment name to run VGGT (use '' to run in current environment)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    blender_dir = Path(args.blender_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms_path = blender_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"{transforms_path} not found")

    transforms = json.loads(transforms_path.read_text())
    print(f"Loaded transforms.json with {len(transforms.get('frames', []))} frames")

    metadata = copy_frames_to_dataset(blender_dir, output_dir, transforms)
    print(f"Copied data for {len(metadata)} cameras to {output_dir}")

    all_frames = gather_all_frames(metadata)
    per_camera_frames = select_first_frame_per_camera(metadata)

    point_cloud_path = output_dir / "points3D_multipleview.ply"

    if not args.skip_vggt:
        if args.vggt_script is None:
            raise ValueError("--vggt_script must be specified when VGGT is enabled")

        tmp_scene_dir = output_dir / "tmp_colmap"
        prepare_vggt_scene(per_camera_frames, tmp_scene_dir)
        conda_env = (
            args.vggt_conda_env.strip() if args.vggt_conda_env is not None else None
        )
        if conda_env == "":
            conda_env = None
        run_vggt_demo(args.vggt_script, tmp_scene_dir, conda_env)

        vggt_points, vggt_colors, pose_by_name = load_vggt_outputs(tmp_scene_dir)

        vggt_centers = []
        blender_centers = []
        for frame in per_camera_frames:
            if frame.tmp_basename is None:
                raise RuntimeError("Temporary VGGT image name missing")
            if frame.tmp_basename not in pose_by_name:
                raise RuntimeError(
                    f"VGGT reconstruction missing pose for {frame.tmp_basename}"
                )
            R_wc, t_wc = pose_by_name[frame.tmp_basename]
            vggt_centers.append(camera_center_from_colmap(R_wc, t_wc))
            c2w = blender_matrix_to_opencv_c2w(frame.transform_matrix)
            blender_centers.append(c2w[:3, 3])

        aligned_points = align_point_cloud(
            vggt_points,
            np.stack(vggt_centers),
            np.stack(blender_centers),
        )

        aligned_points, aligned_colors = maybe_downsample(
            aligned_points, vggt_colors, args.max_point_cloud_points
        )
        write_ply(point_cloud_path, aligned_points, aligned_colors)

        if not args.keep_tmp:
            shutil.rmtree(tmp_scene_dir, ignore_errors=True)
    else:
        if not point_cloud_path.exists():
            raise FileNotFoundError(
                "points3D_multipleview.ply missing; run without --skip_vggt first"
            )

    points = PlyData.read(str(point_cloud_path))["vertex"]
    point_positions = np.stack([points["x"], points["y"], points["z"]], axis=1)
    point_colors = np.stack(
        [points["red"], points["green"], points["blue"]], axis=1
    ).astype(np.uint8)

    sparse_dir = output_dir / "sparse_"
    create_reconstruction(
        all_frames, transforms, sparse_dir, point_positions, point_colors
    )

    compute_poses_bounds_from_frames(
        all_frames,
        point_positions,
        transforms,
        output_dir / "poses_bounds_multipleview.npy",
    )

    dataset_name = args.dataset_name or output_dir.name
    create_config_file(output_dir, dataset_name)

    print("Conversion complete. Dataset ready at:", output_dir)
    print("To train 4DGaussians:")
    print("  cd", REPO_ROOT)
    print(
        "  python train.py -s",
        output_dir,
        f'--port 6017 --expname "multipleview/{dataset_name}" --configs arguments/multipleview/{dataset_name}.py',
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
