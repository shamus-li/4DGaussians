#!/usr/bin/env python3
"""Convert a directory of multi-view videos into the 4DGaussians data format.

The script assumes each video captures a static camera stream (constant pose)
for the entire sequence. It extracts frames from every video, estimates the
camera poses with VGGT using only the first frame per camera (followed by
bundle adjustment), and assembles the assets required by 4DGaussians. When no
explicit output path is provided the dataset is written to
``data/multipleview/<video-folder-name>``.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


MEGASAM_DIR_NAME = "megasam"


def ensure_cudss_accessible() -> None:
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return

    cudss_dirs = sorted(Path(prefix).glob("opt/libcudss-linux-x86_64-*_cuda12-archive"))
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
        except OSError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to preload cuDSS library at {libcudss}: {exc}"
            ) from exc


ensure_cudss_accessible()

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scene.colmap_loader import (  # noqa: E402
    qvec2rotmat,
    read_extrinsics_binary,
    rotmat2qvec,
)
from utils.config_templates import generate_multiview_config  # noqa: E402

SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from scripts.colmap_converter import (  # noqa: E402
    Camera as ColmapCamera,
)
from scripts.colmap_converter import (  # noqa: E402
    Image as ColmapImage,
)
from scripts.colmap_converter import (  # noqa: E402
    Point3D as ColmapPoint3D,
)
from scripts.colmap_converter import (  # noqa: E402
    read_cameras_binary,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
    write_points3D_binary,
    write_points3D_text,
)

SUPPORTED_VIDEO_EXTS = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".webm",
)



@dataclass
class RawFrame:
    index: int
    path: Path
    width: int
    height: int


@dataclass
class CameraSequence:
    name: str
    video_path: Path
    frames: List[RawFrame]
    tmp_image_name: Optional[str] = None


@dataclass
class FrameRecord:
    camera_name: str
    frame_index: int
    image_path: Path
    width: int
    height: int
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    c2w: np.ndarray


class MegaSamInitializer:
    def __init__(self, args: argparse.Namespace, dataset_dir: Path) -> None:
        self.root = args.megasam_root.expanduser().resolve()
        self.env = (args.megasam_env or "megasam").strip()
        self.depth_ckpt = self._resolve_with_root(args.megasam_depth_anything_ckpt)
        self.tracking_weights = self._resolve_with_root(args.megasam_tracking_weights)
        self.raft_ckpt = self._resolve_with_root(args.megasam_raft_ckpt)
        self.da_encoder = args.megasam_da_encoder
        self.dataset_dir = dataset_dir
        self.storage_root = dataset_dir / MEGASAM_DIR_NAME
        self.depth_root = self.storage_root / "Depth-Anything"
        self.unidepth_root = self.storage_root / "UniDepth"
        self.outputs_root = self.storage_root / "outputs_cvd"
        self.depth_root.mkdir(parents=True, exist_ok=True)
        self.unidepth_root.mkdir(parents=True, exist_ok=True)
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.env_vars = os.environ.copy()
        unidepth_path = str(self.root / "UniDepth")
        existing_py_path = self.env_vars.get("PYTHONPATH", "")
        if existing_py_path:
            if unidepth_path not in existing_py_path.split(":"):
                self.env_vars["PYTHONPATH"] = f"{existing_py_path}:{unidepth_path}"
        else:
            self.env_vars["PYTHONPATH"] = unidepth_path

    def _resolve_with_root(self, path: Path) -> Path:
        expanded = path.expanduser()
        if expanded.is_absolute():
            return expanded
        return (self.root / expanded).resolve()

    def _run_conda_script(self, script: str, extra_args: Sequence[str]) -> None:
        if self.env:
            cmd = [
                "conda",
                "run",
                "-n",
                self.env,
                "python",
                script,
                *extra_args,
            ]
        else:
            cmd = [sys.executable, script, *extra_args]
        pretty = " ".join(str(x) for x in cmd)
        print("->", pretty)
        subprocess.run(cmd, cwd=str(self.root), check=True, env=self.env_vars)

    def run_sequence(self, sequence: "CameraSequence") -> Path:
        camera_name = sequence.name
        frame_dir = sequence.frames[0].path.parent
        depth_scene_dir = self.depth_root / camera_name
        if depth_scene_dir.exists():
            shutil.rmtree(depth_scene_dir)
        depth_scene_dir.mkdir(parents=True, exist_ok=True)
        unidepth_scene_dir = self.unidepth_root / camera_name
        if unidepth_scene_dir.exists():
            shutil.rmtree(unidepth_scene_dir)
        unidepth_scene_dir.mkdir(parents=True, exist_ok=True)

        self._run_depth_anything(frame_dir, depth_scene_dir)
        self._run_unidepth(frame_dir, camera_name)
        self._run_camera_tracking(frame_dir, camera_name)
        self._run_flow(frame_dir, camera_name)
        npz_path = self._run_cvd(camera_name)
        self._cleanup_intermediate(camera_name)
        return npz_path

    def _run_depth_anything(self, frame_dir: Path, out_dir: Path) -> None:
        args = [
            "Depth-Anything/run_videos.py",
            "--encoder",
            self.da_encoder,
            "--load-from",
            str(self.depth_ckpt),
            "--img-path",
            str(frame_dir),
            "--outdir",
            str(out_dir),
        ]
        self._run_conda_script(args[0], args[1:])

    def _run_unidepth(self, frame_dir: Path, scene_name: str) -> None:
        args = [
            "UniDepth/scripts/demo_mega-sam.py",
            "--img-path",
            str(frame_dir),
            "--scene-name",
            scene_name,
            "--outdir",
            str(self.unidepth_root),
        ]
        self._run_conda_script(args[0], args[1:])

    def _run_camera_tracking(self, frame_dir: Path, scene_name: str) -> None:
        args = [
            "camera_tracking_scripts/test_demo.py",
            "--datapath",
            str(frame_dir),
            "--weights",
            str(self.tracking_weights),
            "--scene_name",
            scene_name,
            "--mono_depth_path",
            str(self.depth_root),
            "--metric_depth_path",
            str(self.unidepth_root),
            "--disable_vis",
        ]
        self._run_conda_script(args[0], args[1:])

    def _run_flow(self, frame_dir: Path, scene_name: str) -> None:
        args = [
            "cvd_opt/preprocess_flow.py",
            "--datapath",
            str(frame_dir),
            "--model",
            str(self.raft_ckpt),
            "--scene_name",
            scene_name,
            "--mixed_precision",
        ]
        self._run_conda_script(args[0], args[1:])

    def _run_cvd(self, scene_name: str) -> Path:
        args = [
            "cvd_opt/cvd_opt.py",
            "--scene_name",
            scene_name,
            "--output_dir",
            str(self.outputs_root),
        ]
        self._run_conda_script(args[0], args[1:])
        npz_path = self.outputs_root / f"{scene_name}_sgd_cvd_hr.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"MegaSaM CVD output missing at {npz_path}")
        return npz_path

    def _cleanup_intermediate(self, scene_name: str) -> None:
        recon_dir = self.root / "reconstructions" / scene_name
        cache_dir = self.root / "cache_flow" / scene_name
        output_file = self.root / "outputs" / f"{scene_name}_droid.npz"
        shutil.rmtree(recon_dir, ignore_errors=True)
        shutil.rmtree(cache_dir, ignore_errors=True)
        try:
            output_file.unlink()
        except FileNotFoundError:
            pass


def collect_vggt_metadata(
    sequences: Sequence[CameraSequence],
    images_by_name: Dict[str, ColmapImage],
    cameras: Dict[int, ColmapCamera],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float, float, float]]]:
    cam_to_world: Dict[str, np.ndarray] = {}
    intrinsics: Dict[str, Tuple[float, float, float, float]] = {}
    for seq in sequences:
        if not seq.tmp_image_name:
            raise RuntimeError(f"Missing VGGT temporary image for {seq.name}")
        if seq.tmp_image_name not in images_by_name:
            raise KeyError(
                f"VGGT image {seq.tmp_image_name} not found for camera {seq.name}"
            )
        image = images_by_name[seq.tmp_image_name]
        if image.camera_id not in cameras:
            raise KeyError(
                f"Camera id {image.camera_id} missing for VGGT image {seq.tmp_image_name}"
            )
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera_intrinsics_from_colmap(camera)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c)
        cam_to_world[seq.name] = c2w
        intrinsics[seq.name] = (fx, fy, cx, cy)
    return cam_to_world, intrinsics


def compute_cam_to_ref_transforms(
    cam_to_world: Dict[str, np.ndarray], reference_camera: str
) -> Dict[str, np.ndarray]:
    if reference_camera not in cam_to_world:
        raise KeyError(
            f"Reference camera '{reference_camera}' not found in VGGT metadata"
        )
    ref_c2w = cam_to_world[reference_camera]
    ref_inv = np.linalg.inv(ref_c2w)
    transforms: Dict[str, np.ndarray] = {}
    for name, c2w in cam_to_world.items():
        transforms[name] = ref_inv @ c2w
    return transforms


def build_hybrid_megasam_records(
    sequences: Sequence[CameraSequence],
    reference_sequence: CameraSequence,
    initializer: MegaSamInitializer,
    cam_to_ref: Dict[str, np.ndarray],
    intrinsics: Dict[str, Tuple[float, float, float, float]],
    frame_count: int,
    max_point_cloud_points: int,
    oversample_factor: float,
) -> Tuple[List[FrameRecord], np.ndarray, np.ndarray]:
    if frame_count == 0:
        raise RuntimeError("No frames extracted; cannot run MegaSaM initialization")
    rng = np.random.default_rng(0)
    per_frame_quota = max(
        50,
        int(
            oversample_factor
            * max_point_cloud_points
            / max(frame_count, 1)
        ),
    )

    npz_path = initializer.run_sequence(reference_sequence)
    ref_records, ref_points, ref_colors = consume_megasam_npz(
        reference_sequence, npz_path, per_frame_quota, rng
    )

    if len(ref_records) != frame_count:
        raise RuntimeError(
            "MegaSaM reference run produced a different number of frames than expected"
        )

    all_records: List[FrameRecord] = list(ref_records)
    ref_name = reference_sequence.name
    for seq in sequences:
        if seq.name == ref_name:
            continue
        if seq.name not in cam_to_ref:
            raise KeyError(f"No transform available for camera {seq.name}")
        if seq.name not in intrinsics:
            raise KeyError(f"No intrinsics available for camera {seq.name}")
        transform = cam_to_ref[seq.name]
        fx, fy, cx, cy = intrinsics[seq.name]
        if len(seq.frames) != frame_count:
            raise RuntimeError(
                f"Camera {seq.name} has {len(seq.frames)} frames, expected {frame_count}"
            )
        for idx, raw in enumerate(seq.frames):
            base_record = ref_records[idx]
            c2w = base_record.c2w @ transform
            all_records.append(
                FrameRecord(
                    camera_name=seq.name,
                    frame_index=raw.index,
                    image_path=raw.path,
                    width=raw.width,
                    height=raw.height,
                    fl_x=fx,
                    fl_y=fy,
                    cx=cx,
                    cy=cy,
                    c2w=c2w,
                )
            )

    all_records.sort(key=lambda fr: (fr.frame_index, fr.camera_name))
    return all_records, ref_points, ref_colors


def consume_megasam_npz(
    sequence: CameraSequence,
    npz_path: Path,
    samples_per_frame: int,
    rng: np.random.Generator,
) -> Tuple[List[FrameRecord], np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    depths = data["depths"]
    cam_c2w = data["cam_c2w"]
    intrinsic = data["intrinsic"]
    if intrinsic.ndim == 2:
        intrinsic = np.repeat(intrinsic[None, ...], depths.shape[0], axis=0)
    if depths.shape[0] != len(sequence.frames):
        raise RuntimeError(
            f"MegaSaM produced {depths.shape[0]} frames, expected {len(sequence.frames)}"
        )

    frame_records: List[FrameRecord] = []
    sampled_points: List[np.ndarray] = []
    sampled_colors: List[np.ndarray] = []

    for idx, (raw_frame, depth_map, K) in enumerate(
        zip(sequence.frames, depths, intrinsic)
    ):
        c2w = np.array(cam_c2w[idx], dtype=np.float64)
        width = raw_frame.width
        height = raw_frame.height
        scale_x = width / depth_map.shape[1]
        scale_y = height / depth_map.shape[0]
        fx = float(K[0, 0] * scale_x)
        fy = float(K[1, 1] * scale_y)
        cx = float(K[0, 2] * scale_x)
        cy = float(K[1, 2] * scale_y)

        record = FrameRecord(
            camera_name=sequence.name,
            frame_index=raw_frame.index,
            image_path=raw_frame.path,
            width=width,
            height=height,
            fl_x=fx,
            fl_y=fy,
            cx=cx,
            cy=cy,
            c2w=c2w,
        )
        frame_records.append(record)

        points, colors = sample_points_from_depth(
            record, depth_map, samples_per_frame, rng
        )
        if points.size:
            sampled_points.append(points)
            sampled_colors.append(colors)

    if sampled_points:
        all_points = np.concatenate(sampled_points, axis=0)
        all_colors = np.concatenate(sampled_colors, axis=0)
    else:
        all_points = np.zeros((0, 3), dtype=np.float32)
        all_colors = np.zeros((0, 3), dtype=np.uint8)
    return frame_records, all_points, all_colors


def sample_points_from_depth(
    record: FrameRecord,
    depth_map: np.ndarray,
    sample_quota: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    depth_img = Image.fromarray(depth_map.astype(np.float32), mode="F")
    depth_resized = depth_img.resize((record.width, record.height), Image.BILINEAR)
    depth = np.asarray(depth_resized, dtype=np.float32)
    valid_mask = depth > 1e-3
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    if sample_quota > 0 and valid_indices.size > sample_quota:
        choice = rng.choice(valid_indices.size, sample_quota, replace=False)
        flat_idx = valid_indices[choice]
    else:
        flat_idx = valid_indices

    ys, xs = np.unravel_index(flat_idx, depth.shape)
    zs = depth[ys, xs]
    xs_cam = (xs - record.cx) / record.fl_x * zs
    ys_cam = (ys - record.cy) / record.fl_y * zs
    cam_points = np.stack([xs_cam, ys_cam, zs], axis=1)
    world = (record.c2w[:3, :3] @ cam_points.T + record.c2w[:3, 3:4]).T

    with Image.open(record.image_path) as pil_image:
        rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    colors = rgb[ys, xs]
    return world.astype(np.float32), colors


def collect_video_files(root: Path, recursive: bool) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Video directory {root} not found")

    if recursive:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS
        ]
    else:
        files = [
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTS
        ]

    files.sort()
    if not files:
        raise RuntimeError(
            f"No supported video files found in {root} (recursive={recursive})"
        )
    return files


def save_frame(frame_array: np.ndarray, dst_path: Path) -> Tuple[int, int]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(frame_array)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")
    image.save(dst_path, format="PNG")
    width, height = image.size
    return width, height


def extract_frames(
    video_path: Path,
    camera_dir: Path,
    frame_stride: int,
    max_frames: Optional[int],
) -> List[RawFrame]:
    reader = imageio.get_reader(str(video_path))
    frames: List[RawFrame] = []
    saved_index = 0

    try:
        for frame_idx, frame_array in enumerate(reader):
            if frame_idx % frame_stride != 0:
                continue
            saved_index += 1
            if max_frames is not None and saved_index > max_frames:
                break
            dst_path = camera_dir / f"frame_{saved_index:05d}.png"
            width, height = save_frame(frame_array, dst_path)
            frames.append(
                RawFrame(index=saved_index, path=dst_path, width=width, height=height)
            )
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    return frames


def harmonize_frame_counts(sequences: Sequence[CameraSequence]) -> int:
    counts = [len(seq.frames) for seq in sequences]
    if not counts:
        raise ValueError("No camera sequences available")

    target = min(counts)
    if any(count != target for count in counts):
        details = ", ".join(f"{seq.name}:{len(seq.frames)}" for seq in sequences)
        print(
            "WARNING: trimming frame counts to match shortest sequence;"
            f" target={target}. Counts: {details}"
        )
        for seq in sequences:
            while len(seq.frames) > target:
                frame = seq.frames.pop()
                try:
                    frame.path.unlink()
                except FileNotFoundError:
                    pass
    return target


def prepare_vggt_scene(sequences: Iterable[CameraSequence], scene_dir: Path) -> None:
    images_dir = scene_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    for seq in sequences:
        ref_frame = seq.frames[0]
        tmp_name = f"{seq.name}_{ref_frame.path.name}"
        shutil.copy2(ref_frame.path, images_dir / tmp_name)
        seq.tmp_image_name = tmp_name


def run_vggt_pipeline(
    demo_script: Path,
    scene_dir: Path,
    conda_env: Optional[str],
    conf_threshold: float,
    vis_threshold: float,
    min_inliers: int,
    run_bundle_adjust: bool = True,
) -> None:
    if not demo_script.exists():
        raise FileNotFoundError(f"VGGT script not found at {demo_script}")

    if run_bundle_adjust:
        stage = "both"
        extra_args = [
            "--use_ba",
            "--conf_thres_value",
            str(conf_threshold),
            "--vis_thresh",
            str(vis_threshold),
            "--min_inlier_per_frame",
            str(min_inliers),
        ]
    else:
        stage = "vggt"
        extra_args = ["--conf_thres_value", str(conf_threshold)]

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
            stage,
            *extra_args,
        ]
    else:
        cmd = [
            sys.executable,
            str(demo_script),
            "--scene_dir",
            str(scene_dir),
            "--stage",
            stage,
            *extra_args,
        ]
    print("->", " ".join(cmd))
    result = subprocess.run(cmd, cwd=demo_script.parent, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"VGGT script failed during stage '{stage}'")


def load_vggt_outputs(
    scene_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, ColmapCamera], Dict[str, ColmapImage]]:
    sparse_dir = scene_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"VGGT sparse directory missing: {sparse_dir}")

    points_path = sparse_dir / "points.ply"
    if not points_path.exists():
        raise FileNotFoundError(f"VGGT point cloud not found at {points_path}")

    ply = PlyData.read(str(points_path))
    vertices = ply["vertex"]
    points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(
        np.float64
    )
    colors = np.stack(
        [vertices["red"], vertices["green"], vertices["blue"]], axis=1
    ).astype(np.uint8)

    cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    extrinsics = read_extrinsics_binary(sparse_dir / "images.bin")
    images_by_name = {Path(image.name).name: image for image in extrinsics.values()}
    return points, colors, cameras, images_by_name


def camera_intrinsics_from_colmap(
    camera: ColmapCamera,
) -> Tuple[float, float, float, float]:
    model = camera.model.upper()
    params = camera.params
    four_param_models = {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "THIN_PRISM_FISHEYE",
    }
    three_param_models = {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL",
        "FOV",
    }

    if model in four_param_models:
        if params.size < 4:
            raise ValueError(f"Camera model {model} expects at least 4 params")
        fx, fy, cx, cy = params[:4]
    elif model in three_param_models:
        if params.size < 3:
            raise ValueError(f"Camera model {model} expects at least 3 params")
        fx = fy = params[0]
        cx = params[1]
        cy = params[2]
    else:
        raise NotImplementedError(f"Unsupported COLMAP camera model: {model}")
    return float(fx), float(fy), float(cx), float(cy)


def build_frame_records(
    sequences: Iterable[CameraSequence],
    cameras: Dict[int, ColmapCamera],
    images_by_name: Dict[str, ColmapImage],
) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for seq in sequences:
        if not seq.tmp_image_name:
            raise RuntimeError(f"Temporary VGGT image missing for {seq.name}")
        if seq.tmp_image_name not in images_by_name:
            available = ", ".join(images_by_name.keys())
            raise KeyError(
                f"No VGGT pose found for {seq.tmp_image_name}. Available: {available}"
            )
        image = images_by_name[seq.tmp_image_name]
        if image.camera_id not in cameras:
            raise KeyError(f"Camera id {image.camera_id} not found in COLMAP cameras")
        camera = cameras[image.camera_id]
        fx, fy, cx, cy = camera_intrinsics_from_colmap(camera)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3] = image.tvec
        c2w = np.linalg.inv(w2c)

        for raw in seq.frames:
            if raw.width != camera.width or raw.height != camera.height:
                raise ValueError(
                    f"Resolution mismatch for {seq.name}: video frame {raw.width}x{raw.height},"
                    f" COLMAP camera expects {camera.width}x{camera.height}"
                )
            records.append(
                FrameRecord(
                    camera_name=seq.name,
                    frame_index=raw.index,
                    image_path=raw.path,
                    width=raw.width,
                    height=raw.height,
                    fl_x=fx,
                    fl_y=fy,
                    cx=cx,
                    cy=cy,
                    c2w=c2w,
                )
            )
    records.sort(key=lambda fr: (fr.frame_index, fr.camera_name))
    return records


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
        # Gradually increase voxel size until we reach the desired budget.
        while np.asarray(pcd.points).shape[0] > max_points:
            pcd = pcd.voxel_down_sample(voxel)
            voxel *= 1.2
        points = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
        return points, colors
    except ImportError:
        idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        return points[idx], colors[idx]


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


def create_reconstruction(
    frames: Sequence[FrameRecord],
    sparse_dir: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)

    camera_id_map: Dict[str, int] = {}
    cameras_dict: Dict[int, ColmapCamera] = {}
    for frame in frames:
        if frame.camera_name in camera_id_map:
            continue
        width, height = frame.width, frame.height
        fx = frame.fl_x
        fy = frame.fl_y
        cx = frame.cx
        cy = frame.cy

        if abs(fx - fy) > 1e-6:
            model = "PINHOLE"
            params = np.array([fx, fy, cx, cy], dtype=np.float64)
        else:
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
        w2c = np.linalg.inv(frame.c2w)
        R_wc = w2c[:3, :3]
        t_wc = w2c[:3, 3]
        qvec = rotmat2qvec(R_wc).astype(np.float64)
        tvec = t_wc.astype(np.float64)
        rel_name = str(Path(frame.camera_name) / frame.image_path.name)
        images_dict[image_id] = ColmapImage(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id_map[frame.camera_name],
            name=rel_name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros(0, dtype=np.int64),
        )

    points_dict: Dict[int, ColmapPoint3D] = {}
    if points.size > 0:
        sample_count = min(len(points), 5000)
        sample_idx = (
            np.linspace(0, len(points) - 1, sample_count).astype(np.int64)
            if len(points) > sample_count
            else np.arange(len(points), dtype=np.int64)
        )
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


def compute_poses_bounds(
    frames: Sequence[FrameRecord],
    points: np.ndarray,
    output_path: Path,
) -> None:
    data = []
    point_cloud = points.T if points.size > 0 else None
    for frame in frames:
        c2w = frame.c2w
        pose = np.zeros((3, 5), dtype=np.float32)
        pose[:, :3] = c2w[:3, :3]
        pose[:, 3] = c2w[:3, 3]
        pose[:, 4] = np.array([frame.height, frame.width, frame.fl_x], dtype=np.float32)

        w2c = np.linalg.inv(c2w)
        if point_cloud is not None:
            depths = (w2c[:3, :3] @ point_cloud + w2c[:3, 3:4])[2]
            positive = depths[depths > 0]
            if positive.size == 0:
                near, far = 0.1, 10.0
            else:
                near = float(np.percentile(positive, 1) * 0.9)
                far = float(np.percentile(positive, 99) * 1.1)
        else:
            near, far = 0.1, 10.0

        data.append(
            np.concatenate([pose.reshape(-1), np.array([near, far], dtype=np.float32)])
        )

    np.save(output_path, np.stack(data))


def create_config_file(
    output_dir: Path, dataset_name: str, camera_count: int, frame_count: int
) -> None:
    config_path = output_dir / "config.py"

    config_body = generate_multiview_config(
        camera_count, frame_count, dataset_name=dataset_name
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(config_body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert multi-view videos to the 4DGaussians multi-view dataset format",
    )
    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="Directory containing the input videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory for the generated dataset (default:"
            " data/multipleview/<input-folder-name>)"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name for configuration export (default: output directory name)",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Sample every Nth frame from the videos (default: 1)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Limit the number of frames extracted per video after stride (default: all)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for videos under --videos",
    )
    parser.add_argument(
        "--max_point_cloud_points",
        type=int,
        default=300000,
        help="Maximum number of points to keep in the exported point cloud",
    )
    parser.add_argument(
        "--init_method",
        choices=("vggt", "megasam"),
        default="vggt",
        help="Initializer to use for pose/point estimation",
    )
    parser.add_argument(
        "--megasam_root",
        type=Path,
        default=Path("../mega-sam"),
        help="Path to the MegaSaM repository",
    )
    parser.add_argument(
        "--megasam_env",
        type=str,
        default="megasam",
        help="Conda environment name for MegaSaM (empty string -> current env)",
    )
    parser.add_argument(
        "--megasam_depth_anything_ckpt",
        type=Path,
        default=Path("Depth-Anything/checkpoints/depth_anything_vitl14.pth"),
        help="DepthAnything checkpoint path relative to MegaSaM root unless absolute",
    )
    parser.add_argument(
        "--megasam_tracking_weights",
        type=Path,
        default=Path("checkpoints/megasam_final.pth"),
        help="Camera tracking weights path (relative to MegaSaM root unless absolute)",
    )
    parser.add_argument(
        "--megasam_raft_ckpt",
        type=Path,
        default=Path("cvd_opt/raft-things.pth"),
        help="RAFT checkpoint path (relative to MegaSaM root unless absolute)",
    )
    parser.add_argument(
        "--megasam_da_encoder",
        type=str,
        default="vitl",
        help="DepthAnything encoder variant (vits/vitb/vitl)",
    )
    parser.add_argument(
        "--megasam_reference_camera",
        type=str,
        default=None,
        help="Camera name to run MegaSaM on (default: first camera)",
    )
    parser.add_argument(
        "--megasam_point_oversample",
        type=float,
        default=2.0,
        help="Oversample factor before downsampling MegaSaM point clouds",
    )
    parser.add_argument(
        "--vggt_script",
        type=Path,
        default=Path("~/repos/vggt/demo_colmap.py"),
        help="Path to VGGT demo_colmap.py",
    )
    parser.add_argument(
        "--vggt_conda_env",
        type=str,
        default=None,
        help="Conda env name for running VGGT (empty string -> current environment)",
    )
    parser.add_argument(
        "--vggt_conf_threshold",
        type=float,
        default=3.0,
        help="Confidence threshold passed to VGGT (default: 3.0)",
    )
    parser.add_argument(
        "--vggt_vis_threshold",
        type=float,
        default=0.05,
        help="Visibility threshold for VGGT tracks during BA (default: 0.05)",
    )
    parser.add_argument(
        "--vggt_min_inliers",
        type=int,
        default=16,
        help="Minimum inliers per frame required by VGGT BA (default: 16)",
    )
    parser.add_argument(
        "--keep_tmp",
        action="store_true",
        help="Keep the temporary VGGT scene directory inside the output",
    )
    parser.add_argument(
        "--legacy_naming",
        action="store_true",
        help="Use legacy cam_XXXXX naming instead of preserving video filenames",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    video_root = args.videos.expanduser().resolve()
    if args.output is None:
        default_output = REPO_ROOT / "data" / "multipleview" / video_root.name
        output_dir = default_output.resolve()
    else:
        output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset_name or output_dir.name

    # Sentinels to coordinate multi-task pipelines
    in_progress_path = output_dir / ".conversion_in_progress"
    done_path = output_dir / ".conversion_done"
    # Clear any stale 'done' file and mark conversion as in-progress atomically
    try:
        if done_path.exists():
            done_path.unlink()
    except Exception:
        pass
    tmp_inp = in_progress_path.with_suffix(in_progress_path.suffix + ".tmp")
    try:
        tmp_inp.write_text(
            json.dumps({"ok": False, "ts": datetime.now().isoformat()}),
            encoding="utf-8",
        )
        tmp_inp.replace(in_progress_path)
    except Exception:
        # Non-fatal; continue conversion even if sentinel cannot be written
        pass

    success = False
    
    try:
        video_files = collect_video_files(video_root, recursive=args.recursive)
        print(f"Found {len(video_files)} video(s)")

        sequences: List[CameraSequence] = []
        for idx, video_path in enumerate(video_files):
            if args.legacy_naming:
                camera_name = f"cam_{idx + 1:05d}"
            else:
                # Use video filename (without extension) as camera name
                original_stem = video_path.stem
                # Sanitize: replace spaces/hyphens with underscores, keep only ASCII alphanumeric
                video_stem = original_stem.replace(" ", "_").replace("-", "_")
                # Keep only ASCII letters, numbers, and underscores
                video_stem = "".join(c if (c.isalnum() and ord(c) < 128) or c == "_" else "_" for c in video_stem)
                # Remove consecutive underscores and strip leading/trailing underscores
                import re
                video_stem = re.sub(r"_+", "_", video_stem).strip("_")
                # Fallback to generic name if sanitization results in empty string
                if not video_stem:
                    video_stem = "camera"
                    print(f"WARNING: Video name '{original_stem}' could not be sanitized, using '{video_stem}'")
                # Limit length to avoid filesystem issues (max 50 chars for base name)
                if len(video_stem) > 50:
                    video_stem = video_stem[:50].rstrip("_")
                    print(f"WARNING: Video name '{original_stem}' truncated to '{video_stem}'")
                camera_name = f"{video_stem}_{idx + 1:03d}"

            camera_dir = output_dir / camera_name
            frames = extract_frames(
                video_path,
                camera_dir,
                frame_stride=max(1, args.frame_stride),
                max_frames=args.max_frames,
            )
            sequences.append(
                CameraSequence(name=camera_name, video_path=video_path, frames=frames)
            )
            print(
                f"Extracted {len(frames)} frame(s) from {video_path.name} -> {camera_dir.relative_to(output_dir)}"
            )

        frame_count = harmonize_frame_counts(sequences)
        print(f"All cameras will use {frame_count} frame(s)")

        reference_sequence: Optional[CameraSequence] = None
        if args.init_method == "megasam":
            if not sequences:
                raise RuntimeError("No cameras available for MegaSaM initialization")
            if args.megasam_reference_camera:
                target = args.megasam_reference_camera.strip()
                reference_sequence = next(
                    (seq for seq in sequences if seq.name == target), None
                )
                if reference_sequence is None:
                    raise ValueError(
                        f"Reference camera '{target}' not found. Available: {[seq.name for seq in sequences]}"
                    )
            else:
                def select_by_keyword(keywords: Sequence[str]) -> Optional[CameraSequence]:
                    for keyword in keywords:
                        keyword_lower = keyword.lower()
                        for seq in sequences:
                            if keyword_lower in seq.name.lower():
                                return seq
                    return None

                reference_sequence = select_by_keyword(["wide"])
                if reference_sequence is None:
                    reference_sequence = select_by_keyword(["right"])
                if reference_sequence is None:
                    reference_sequence = sequences[0]
            print(f"MegaSaM reference camera: {reference_sequence.name}")

        tmp_scene_dir = output_dir / "tmp_vggt"
        if tmp_scene_dir.exists():
            shutil.rmtree(tmp_scene_dir)
        tmp_scene_dir.mkdir(parents=True, exist_ok=True)

        prepare_vggt_scene(sequences, tmp_scene_dir)

        conda_env = (
            args.vggt_conda_env.strip() if args.vggt_conda_env is not None else None
        )
        if conda_env == "":
            conda_env = None

        vggt_script = args.vggt_script.expanduser().resolve()
        run_vggt_pipeline(
            vggt_script,
            tmp_scene_dir,
            conda_env=conda_env,
            conf_threshold=args.vggt_conf_threshold,
            vis_threshold=args.vggt_vis_threshold,
            min_inliers=max(1, args.vggt_min_inliers),
            run_bundle_adjust=True,
        )

        points, colors, cameras, images_by_name = load_vggt_outputs(tmp_scene_dir)
        if points.size == 0 and args.init_method == "vggt":
            raise RuntimeError("VGGT did not produce any 3D points. Aborting.")

        if not args.keep_tmp:
            shutil.rmtree(tmp_scene_dir, ignore_errors=True)

        if args.init_method == "megasam":
            assert reference_sequence is not None
            cam_to_world, intr_map = collect_vggt_metadata(
                sequences, images_by_name, cameras
            )
            cam_to_ref = compute_cam_to_ref_transforms(
                cam_to_world, reference_sequence.name
            )
            initializer = MegaSamInitializer(args, output_dir)
            frames, raw_points, raw_colors = build_hybrid_megasam_records(
                sequences,
                reference_sequence,
                initializer,
                cam_to_ref,
                intr_map,
                frame_count,
                args.max_point_cloud_points,
                args.megasam_point_oversample,
            )
            print(
                f"Prepared {len(frames)} hybrid MegaSaM frame records across {len(sequences)} cameras"
            )
        else:
            frames = build_frame_records(sequences, cameras, images_by_name)
            print(
                f"Prepared {len(frames)} frame records across {len(sequences)} cameras"
            )
            raw_points, raw_colors = points, colors

        down_points, down_colors = maybe_downsample(
            raw_points, raw_colors, args.max_point_cloud_points
        )
        point_cloud_path = output_dir / "points3D_multipleview.ply"
        write_ply(point_cloud_path, down_points, down_colors)
        print(
            f"Saved point cloud with {down_points.shape[0]} points -> {point_cloud_path}"
        )

        sparse_dir_train = output_dir / "sparse_train"
        create_reconstruction(frames, sparse_dir_train, down_points, down_colors)
        print(f"Wrote COLMAP artifacts to {sparse_dir_train}")

        legacy_sparse_dir = output_dir / "sparse_"
        shutil.copytree(sparse_dir_train, legacy_sparse_dir, dirs_exist_ok=True)
        print(f"Duplicated sparse data for legacy path -> {legacy_sparse_dir}")

        poses_bounds_path = output_dir / "poses_bounds_multipleview.npy"
        compute_poses_bounds(frames, down_points, poses_bounds_path)
        print(f"Wrote poses & bounds -> {poses_bounds_path}")

        create_config_file(output_dir, dataset_name, len(sequences), frame_count)
        print(f"Generated training config at {output_dir / 'config.py'}")

        # Write conversion done sentinel atomically with metadata
        try:
            done_meta = {
                "ok": True,
                "ts": datetime.now().isoformat(),
                "dataset": dataset_name,
                "cameras": len(sequences),
                "frames_per_camera": int(frame_count),
                "total_frames": int(len(sequences) * frame_count),
                "generator": "video_to_4dgs.py",
            }
            tmp_done = done_path.with_suffix(done_path.suffix + ".tmp")
            tmp_done.write_text(json.dumps(done_meta), encoding="utf-8")
            tmp_done.replace(done_path)
        except Exception:
            # Non-fatal; training can still proceed without metadata
            pass
        success = True

        print("Conversion complete. Dataset ready at:", output_dir)
        print("To train 4DGaussians:")
        print("  cd", REPO_ROOT)
        print(
            "  python train.py -s",
            output_dir,
            f'--expname "multipleview/{dataset_name}"',
        )
        print("\n(config.py will be auto-detected from the dataset directory)")
        return 0
    finally:
        # Always clear in-progress sentinel at the end
        try:
            if in_progress_path.exists():
                in_progress_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
