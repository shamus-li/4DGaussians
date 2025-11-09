#!/usr/bin/env python3
"""Convert Blender multi-camera outputs to the 4DGaussians multi-view format.

This variant relies on VGGT (demo_colmap.py) to generate an aligned point cloud
while using the ground-truth Blender camera poses for all COLMAP artifacts.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


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
        except OSError as exc:
            raise RuntimeError(
                f"Failed to preload cuDSS library at {libcudss}: {exc}"
            ) from exc


ensure_cudss_accessible()

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary  # noqa: E402
from utils.config_templates import generate_multiview_config  # noqa: E402

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
    fl_x: float
    fl_y: float
    cx: float
    cy: float
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


def convert_image_to_png(src: Path, dst: Path) -> Tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        width, height = img.size
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")
        img.save(dst, format="PNG")
    return width, height


def copy_frames_to_dataset(
    blender_dir: Path, output_dir: Path, transforms: Dict, camera_id_offset: int = 0
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
        canonical_cam = f"cam_{camera_id_offset + cam_idx + 1:05d}"
        cam_dir = output_dir / canonical_cam
        cam_dir.mkdir(parents=True, exist_ok=True)

        ordered_frames = sorted(
            grouped[cam_name], key=lambda f: parse_frame_index(f["file_path"])
        )
        frame_records: List[FrameRecord] = []

        for seq_idx, frame in enumerate(ordered_frames):
            src_path = resolve_source_image(blender_dir, frame["file_path"])
            dst_path = cam_dir / f"frame_{seq_idx + 1:05d}.png"
            width, height = convert_image_to_png(src_path, dst_path)

            # Extract per-frame focal lengths, fallback to global transforms
            if "fl_x" in frame and "fl_y" in frame:
                fl_x = float(frame["fl_x"])
                fl_y = float(frame["fl_y"])
                cx = float(frame.get("cx", width / 2))
                cy = float(frame.get("cy", height / 2))
            elif "fl_x" in transforms and "fl_y" in transforms:
                fl_x = float(transforms["fl_x"])
                fl_y = float(transforms["fl_y"])
                cx = float(transforms.get("cx", width / 2))
                cy = float(transforms.get("cy", height / 2))
            elif "camera_angle_x" in frame:
                fl_x = width / (2 * math.tan(float(frame["camera_angle_x"]) / 2))
                fl_y = fl_x
                cx = width / 2
                cy = height / 2
            elif "camera_angle_x" in transforms:
                fl_x = width / (2 * math.tan(float(transforms["camera_angle_x"]) / 2))
                fl_y = fl_x
                cx = width / 2
                cy = height / 2
            else:
                raise ValueError(
                    f"Unable to determine focal length for frame {frame['file_path']}"
                )

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
                    fl_x=fl_x,
                    fl_y=fl_y,
                    cx=cx,
                    cy=cy,
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


def downsample_vggt_frames(
    frames: List[FrameRecord], max_count: Optional[int]
) -> List[FrameRecord]:
    if max_count is None or max_count <= 0 or len(frames) <= max_count:
        return list(frames)

    step = len(frames) / float(max_count)
    selected_indices: List[int] = []
    seen: set[int] = set()

    for i in range(max_count):
        idx = min(int(round(i * step)), len(frames) - 1)
        if idx not in seen:
            selected_indices.append(idx)
            seen.add(idx)

    if len(selected_indices) < max_count:
        for idx in range(len(frames)):
            if idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)
                if len(selected_indices) == max_count:
                    break

    selected_indices.sort()
    return [frames[idx] for idx in selected_indices]


def prepare_vggt_scene(
    frames: Iterable[FrameRecord],
    scene_dir: Path,
    max_cameras: Optional[int] = None,
) -> List[FrameRecord]:
    images_dir = scene_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    frame_list = list(frames)
    for frame in frame_list:
        frame.tmp_basename = None

    selected_frames = downsample_vggt_frames(frame_list, max_cameras)
    if max_cameras and len(frame_list) > len(selected_frames):
        print(
            f"Downsampling VGGT inputs from {len(frame_list)} to {len(selected_frames)} cameras to reduce memory"
        )

    for frame in selected_frames:
        basename = f"{frame.camera_name}_{frame.dst_path.name}"
        shutil.copy2(frame.dst_path, images_dir / basename)
        frame.tmp_basename = basename

    return selected_frames


def run_vggt_pipeline(
    demo_script: Path,
    scene_dir: Path,
    conda_env: Optional[str],
    conf_threshold: float,
    vis_threshold: float,
    min_inliers: int,
    run_bundle_adjust: bool = False,
    query_frame_num: Optional[int] = None,
    max_query_points: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> str:
    if not demo_script.exists():
        raise FileNotFoundError(f"VGGT script not found at {demo_script}")

    def build_args(enable_ba: bool) -> Tuple[str, List[str]]:
        if enable_ba:
            stage_name = "both"
            arg_list = [
                "--use_ba",
                "--conf_thres_value",
                str(conf_threshold),
                "--vis_thresh",
                str(vis_threshold),
                "--min_inlier_per_frame",
                str(min_inliers),
            ]
        else:
            stage_name = "vggt"
            arg_list = ["--conf_thres_value", str(conf_threshold)]
        if query_frame_num:
            arg_list.extend(["--query_frame_num", str(query_frame_num)])
        if max_query_points:
            arg_list.extend(["--max_query_pts", str(max_query_points)])
        return stage_name, arg_list

    def run_once(enable_ba: bool) -> Tuple[int, str]:
        stage_name, extra_args = build_args(enable_ba)
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
                stage_name,
                *extra_args,
            ]
        else:
            cmd = [
                sys.executable,
                str(demo_script),
                "--scene_dir",
                str(scene_dir),
                "--stage",
                stage_name,
                *extra_args,
            ]
        print("->", " ".join(cmd))
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        result = subprocess.run(cmd, cwd=demo_script.parent, check=False, env=env)
        return result.returncode, stage_name

    return_code, stage = run_once(run_bundle_adjust)
    if return_code != 0 and run_bundle_adjust:
        print(
            "[WARN] VGGT bundle-adjust stage failed; retrying without BA for "
            f"{scene_dir}"
        )
        return_code, stage = run_once(False)

    if return_code != 0:
        raise RuntimeError(f"VGGT script failed during stage '{stage}'")
    return stage


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

    try:
        from scripts.colmap_converter import (
            Camera as ColmapCamera,
        )
        from scripts.colmap_converter import (
            Image as ColmapImage,
        )
        from scripts.colmap_converter import (
            Point3D as ColmapPoint3D,
        )
        from scripts.colmap_converter import (
            write_cameras_binary,
            write_cameras_text,
            write_images_binary,
            write_images_text,
            write_points3D_binary,
            write_points3D_text,
        )
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(
            "colmap_converter.py is required to export COLMAP artifacts"
        ) from exc

    camera_id_map: Dict[str, int] = {}
    cameras_dict: Dict[int, ColmapCamera] = {}
    for frame in frames:
        if frame.camera_name in camera_id_map:
            continue
        width, height = frame.width, frame.height

        # Use per-frame focal lengths from FrameRecord
        fx = frame.fl_x
        fy = frame.fl_y
        cx = frame.cx
        cy = frame.cy

        # Use PINHOLE if fx != fy, otherwise SIMPLE_PINHOLE
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

        # Use per-frame focal length from FrameRecord
        fx = frame.fl_x

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
        description="Convert Blender multi-cam renders to 4DGaussians multi-view format",
    )
    parser.add_argument(
        "--blender",
        required=True,
        type=Path,
        help="Directory containing Blender outputs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Target dataset directory",
    )
    parser.add_argument(
        "--dataset_name",
        help="Optional dataset name for config generation (defaults to output directory name)",
    )
    parser.add_argument(
        "--test_blender",
        type=Path,
        default=Path(
            "/share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego"
        ).expanduser(),
        help="Directory containing test dataset Blender outputs (default: /share/monakhova/shamus_data/multiplexed_pixels/dnerf/lego)",
    )
    parser.add_argument(
        "--test_transforms",
        type=str,
        default="transforms_val.json",
        help="Name of transforms file for test dataset (default: transforms_val.json)",
    )
    parser.add_argument(
        "--video_blender",
        type=Path,
        help="Directory containing video dataset Blender outputs (defaults to same as --test_blender)",
    )
    parser.add_argument(
        "--video_transforms",
        type=str,
        default="transforms_val.json",
        help="Name of transforms file for video dataset (default: transforms_val.json)",
    )
    parser.add_argument(
        "--vggt_script",
        type=Path,
        default=Path("~/repos/vggt/demo_colmap.py").expanduser(),
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
        default=300000,
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
    parser.add_argument(
        "--vggt_conf_threshold",
        type=float,
        default=3.0,
        help="Confidence threshold for VGGT point cloud generation (default: 3.0, VGGT default: 5.0)",
    )
    parser.add_argument(
        "--vggt_vis_threshold",
        type=float,
        default=0.05,
        help="Visibility threshold for VGGT tracks during bundle adjustment (default: 0.05, ignored unless --enable_bundle_adjust is set)",
    )
    parser.add_argument(
        "--vggt_min_inliers",
        type=int,
        default=16,
        help="Minimum inliers per frame required by VGGT bundle adjustment (default: 16, ignored unless --enable_bundle_adjust is set)",
    )
    parser.add_argument(
        "--enable_bundle_adjust",
        action="store_true",
        help="Run VGGT bundle adjustment after the initial reconstruction (disabled by default)",
    )
    parser.add_argument(
        "--vggt_image_resolution",
        type=int,
        default=1024,
        help="Base square resolution for VGGT input images (default: 1024)",
    )
    parser.add_argument(
        "--vggt_fixed_resolution",
        type=int,
        default=518,
        help="Internal feature-map resolution used by VGGT (default: 518)",
    )
    parser.add_argument(
        "--vggt_query_frame_num",
        type=int,
        default=8,
        help="Value passed to demo_colmap.py --query_frame_num (default: 8)",
    )
    parser.add_argument(
        "--vggt_max_query_points",
        type=int,
        default=4096,
        help="Value passed to demo_colmap.py --max_query_pts (default: 4096)",
    )
    parser.add_argument(
        "--vggt_max_cameras",
        type=int,
        default=0,
        help="Optional cap on number of cameras passed to VGGT (0 keeps all cameras)",
    )
    parser.add_argument(
        "--vggt_skip_tracks",
        action="store_true",
        help="Skip VGGT track prediction entirely to reduce memory usage",
    )
    parser.add_argument(
        "--vggt_low_memory_threshold",
        type=int,
        default=80,
        help="Enable low-memory VGGT settings when camera count exceeds this threshold (0 disables)",
    )
    parser.add_argument(
        "--vggt_low_memory_resolution",
        type=int,
        default=768,
        help="VGGT image resolution to use in low-memory mode",
    )
    parser.add_argument(
        "--vggt_low_memory_query_frames",
        type=int,
        default=4,
        help="Query frame count to use in low-memory VGGT mode",
    )
    parser.add_argument(
        "--vggt_low_memory_query_points",
        type=int,
        default=2048,
        help="Max query points to use in low-memory VGGT mode",
    )
    parser.add_argument(
        "--vggt_low_memory_max_cameras",
        type=int,
        default=96,
        help="Camera cap applied when low-memory VGGT mode is active (0 keeps existing limit)",
    )
    parser.add_argument(
        "--keep_vggt_tracks_low_mem",
        action="store_false",
        dest="vggt_low_memory_skip_tracks",
        help="Keep VGGT track prediction when low-memory mode triggers",
    )
    parser.set_defaults(vggt_low_memory_skip_tracks=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    blender_dir = args.blender.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name or output_dir.name

    # Process training dataset
    transforms_path = blender_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"{transforms_path} not found")

    transforms = json.loads(transforms_path.read_text())
    print(f"Loaded transforms.json with {len(transforms.get('frames', []))} frames")

    metadata = copy_frames_to_dataset(blender_dir, output_dir, transforms)
    print(f"Copied training data for {len(metadata)} cameras to {output_dir}")

    # Process test dataset
    test_blender_dir = args.test_blender.expanduser().resolve()
    test_transforms_path = test_blender_dir / args.test_transforms
    test_metadata = None
    if test_transforms_path.exists():
        test_transforms = json.loads(test_transforms_path.read_text())
        print(
            f"Loaded {args.test_transforms} with {len(test_transforms.get('frames', []))} frames"
        )

        test_metadata = copy_frames_to_dataset(
            test_blender_dir,
            output_dir,
            test_transforms,
            camera_id_offset=len(metadata),
        )
        print(f"Copied test data for {len(test_metadata)} cameras to {output_dir}")
    else:
        print(
            f"Warning: Test transforms not found at {test_transforms_path}, skipping test dataset"
        )

    # Process video dataset
    video_blender_dir = args.video_blender
    if video_blender_dir is None:
        video_blender_dir = test_blender_dir
    else:
        video_blender_dir = video_blender_dir.expanduser().resolve()

    video_transforms_path = video_blender_dir / args.video_transforms
    video_metadata = None
    if video_transforms_path.exists():
        video_transforms = json.loads(video_transforms_path.read_text())
        print(
            f"Loaded {args.video_transforms} with {len(video_transforms.get('frames', []))} frames for video"
        )

        video_metadata = copy_frames_to_dataset(
            video_blender_dir,
            output_dir,
            video_transforms,
            camera_id_offset=len(metadata)
            + (len(test_metadata) if test_metadata else 0),
        )
        print(f"Copied video data for {len(video_metadata)} cameras to {output_dir}")
    else:
        print(
            f"Warning: Video transforms not found at {video_transforms_path}, skipping video dataset"
        )

    all_frames = gather_all_frames(metadata)
    per_camera_frames = select_first_frame_per_camera(metadata)
    vggt_camera_count = len(per_camera_frames)

    point_cloud_path = output_dir / "points3D_multipleview.ply"

    if not args.skip_vggt:
        vggt_script_path = (
            args.vggt_script.expanduser() if args.vggt_script is not None else None
        )
        if vggt_script_path is None:
            raise ValueError("--vggt_script must be specified when VGGT is enabled")

        tmp_scene_dir = output_dir / "tmp_colmap"
        limit_cameras: Optional[int] = (
            args.vggt_max_cameras if args.vggt_max_cameras > 0 else None
        )
        image_resolution = max(
            args.vggt_fixed_resolution, args.vggt_image_resolution
        )
        fixed_resolution = max(64, args.vggt_fixed_resolution)
        query_frame_num = max(1, args.vggt_query_frame_num)
        max_query_points = max(256, args.vggt_max_query_points)
        skip_tracks = bool(args.vggt_skip_tracks)

        low_memory = (
            args.vggt_low_memory_threshold > 0
            and vggt_camera_count > args.vggt_low_memory_threshold
        )
        if low_memory:
            print(
                f"VGGT low-memory mode enabled: {vggt_camera_count} cameras exceed threshold {args.vggt_low_memory_threshold}"
            )
            if args.vggt_low_memory_skip_tracks:
                skip_tracks = True

        if limit_cameras is not None and limit_cameras < 3:
            limit_cameras = 3

        vggt_frames = prepare_vggt_scene(
            per_camera_frames, tmp_scene_dir, max_cameras=limit_cameras
        )
        if not vggt_frames:
            raise RuntimeError("No frames selected for VGGT reconstruction")
        if len(vggt_frames) != vggt_camera_count:
            print(
                f"Using {len(vggt_frames)} of {vggt_camera_count} cameras for VGGT reconstruction"
            )
        conda_env = (
            args.vggt_conda_env.strip() if args.vggt_conda_env is not None else None
        )
        if conda_env == "":
            conda_env = None

        base_env_overrides: Dict[str, str] = {
            "VGGT_IMG_RES": str(image_resolution),
            "VGGT_FIXED_RES": str(fixed_resolution),
        }
        attempt_specs: List[Tuple[float, bool]] = []

        def add_attempt(conf_value: float, skip: bool) -> None:
            value = max(0.0, float(conf_value))
            key = (round(value, 4), skip)
            if key not in attempt_specs:
                attempt_specs.append(key)

        add_attempt(args.vggt_conf_threshold, skip_tracks)
        if args.vggt_conf_threshold > 2.0:
            add_attempt(2.0, skip_tracks)
        if args.vggt_conf_threshold > 1.2:
            add_attempt(1.2, skip_tracks)
        add_attempt(0.6, skip_tracks)
        add_attempt(0.0, skip_tracks)
        if skip_tracks:
            add_attempt(0.0, False)

        vggt_points: np.ndarray | None = None
        vggt_colors: np.ndarray | None = None
        pose_by_name: Dict[str, Tuple[np.ndarray, np.ndarray]] | None = None
        sparse_dir = tmp_scene_dir / "sparse"
        cache_file = tmp_scene_dir / "cache_vggt_result.pt"

        ba_enabled = args.enable_bundle_adjust

        for attempt_idx, (conf_value, attempt_skip_tracks) in enumerate(
            attempt_specs
        ):
            if attempt_idx > 0:
                print(
                    f"Retrying VGGT with conf_thres={conf_value} "
                    f"{'(skip tracks)' if attempt_skip_tracks else '(tracks enabled)'}"
                )

            if sparse_dir.exists():
                shutil.rmtree(sparse_dir, ignore_errors=True)
            if cache_file.exists():
                cache_file.unlink()

            env_overrides = dict(base_env_overrides)
            if attempt_skip_tracks:
                env_overrides["VGGT_SKIP_TRACKS"] = "1"

            stage_used = run_vggt_pipeline(
                vggt_script_path,
                tmp_scene_dir,
                conda_env=conda_env,
                conf_threshold=conf_value,
                vis_threshold=args.vggt_vis_threshold,
                min_inliers=max(1, args.vggt_min_inliers),
                run_bundle_adjust=ba_enabled,
                query_frame_num=query_frame_num,
                max_query_points=max_query_points,
                extra_env=env_overrides,
            )
            print(f"VGGT stage used: {stage_used}")
            if ba_enabled and stage_used != "both":
                print("[WARN] VGGT bundle adjustment unavailable; continuing without BA for remaining attempts.")
                ba_enabled = False

            try:
                vggt_points, vggt_colors, pose_by_name = load_vggt_outputs(
                    tmp_scene_dir
                )
            except FileNotFoundError:
                continue

            if vggt_points.size > 0:
                break

            print(
                f"VGGT run with conf_thres={conf_value} produced 0 points – retrying"
            )

        if vggt_points is None or vggt_points.size == 0 or pose_by_name is None:
            raise RuntimeError(
                "VGGT produced no 3D points even after fallback attempts"
            )

        vggt_centers = []
        blender_centers = []
        for frame in vggt_frames:
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

    # Create separate sparse directories for train, test, and video
    train_frames = gather_all_frames(metadata)
    sparse_dir_train = output_dir / "sparse_train"
    create_reconstruction(
        train_frames, transforms, sparse_dir_train, point_positions, point_colors
    )

    if test_metadata:
        test_frames = gather_all_frames(test_metadata)
        sparse_dir_test = output_dir / "sparse_test"
        create_reconstruction(
            test_frames, test_transforms, sparse_dir_test, point_positions, point_colors
        )

    if video_metadata:
        video_frames = gather_all_frames(video_metadata)
        sparse_dir_video = output_dir / "sparse_video"
        create_reconstruction(
            video_frames,
            video_transforms,
            sparse_dir_video,
            point_positions,
            point_colors,
        )

    # Also create the combined sparse_ for compatibility
    all_frames_combined = (
        train_frames
        + (test_frames if test_metadata else [])
        + (video_frames if video_metadata else [])
    )
    sparse_dir = output_dir / "sparse_"
    create_reconstruction(
        all_frames_combined, transforms, sparse_dir, point_positions, point_colors
    )

    compute_poses_bounds_from_frames(
        train_frames,
        point_positions,
        transforms,
        output_dir / "poses_bounds_multipleview.npy",
    )

    camera_count = len(metadata)
    frame_counts = [len(info["frames"]) for info in metadata.values()]
    frame_count = max(frame_counts) if frame_counts else 0

    create_config_file(output_dir, dataset_name, camera_count, frame_count)

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


if __name__ == "__main__":
    raise SystemExit(main())
