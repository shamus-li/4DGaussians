import os
import numpy as np
from typing import Dict, List
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image, ImageFile, UnidentifiedImageError
import torch
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo, fetchPly
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        # Keep a legacy focal field for backward-compatibility; real per-image
        # intrinsics are tracked via camera_ids + cam_intrinsics.
        any_cam = next(iter(cam_intrinsics.values()))
        self.focal = [any_cam.params[0], any_cam.params[0]]
        self.FovY = focal2fov(self.focal[0], any_cam.height)
        self.FovX = focal2fov(self.focal[0], any_cam.width)
        self.cam_intrinsics = cam_intrinsics
        self.transform = T.ToTensor()
        (
            self.image_paths,
            self.image_poses,
            self.image_times,
            self.camera_ids,
        ) = self.load_images_path(cam_folder, cam_extrinsics, cam_intrinsics, split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
        
    
    def load_images_path(self, cam_folder, cam_extrinsics, cam_intrinsics, split):
        image_paths = []
        image_poses = []
        image_times = []
        camera_centers = []
        camera_ids = []
        cam_root = Path(cam_folder)
        grouped: Dict[str, List] = {}
        for extr in cam_extrinsics.values():
            cam_name = str(Path(extr.name).parent)
            grouped.setdefault(cam_name, []).append(extr)

        for cam_name in sorted(grouped.keys()):
            extr_list = grouped[cam_name]
            extr_list.sort(key=lambda e: Path(e.name).stem)
            image_length = len(extr_list)
            if image_length == 0:
                continue

            extr0 = extr_list[0]
            R0 = np.transpose(qvec2rotmat(extr0.qvec))
            t0 = np.array(extr0.tvec)
            C0 = -R0.transpose() @ t0
            camera_centers.append(C0)

            if split == "test":
                indices = sorted({0, image_length // 3, (2 * image_length) // 3})
            elif split == "all":
                indices = range(image_length)
            else:  # train
                indices = range(image_length)

            for idx in indices:
                extr = extr_list[idx]
                R = np.transpose(qvec2rotmat(extr.qvec))
                T = np.array(extr.tvec)
                image_path = cam_root / extr.name
                image_paths.append(str(image_path))
                image_poses.append((R, T))
                image_times.append(float(idx / max(image_length - 1, 1)))
                camera_ids.append(extr.camera_id)

        self.camera_centers = np.array(camera_centers, dtype=np.float32)
        return image_paths, image_poses, image_times, camera_ids
    
    def get_video_cam_infos(self,datadir):
        """Generate video camera trajectory that smoothly interpolates through input views.

        Maintains viewing directions from the input cameras rather than always looking
        at the scene center.
        """
        N_views = 300

        # Gather camera centers and rotation matrices from input cameras
        centers = getattr(self, "camera_centers", None)
        input_rotations = []

        # Collect rotation matrices from image_poses
        grouped_poses = {}
        for img_path, (R, T) in zip(self.image_paths, self.image_poses):
            cam_name = str(Path(img_path).parent.name)
            if cam_name not in grouped_poses:
                grouped_poses[cam_name] = []
            grouped_poses[cam_name].append((R, T))

        # Get representative pose for each camera (use temporal average)
        camera_Rs = []
        camera_centers_computed = []
        for cam_name in sorted(grouped_poses.keys()):
            poses = grouped_poses[cam_name]
            # Use the middle pose as representative
            mid_idx = len(poses) // 2
            R, T = poses[mid_idx]
            camera_Rs.append(R)
            # Compute camera center: C = -R^T @ T
            C = -R.transpose() @ T
            camera_centers_computed.append(C)

        camera_Rs = np.array(camera_Rs)
        camera_centers_computed = np.array(camera_centers_computed)

        # Use computed centers if stored centers are not available
        if centers is None or len(centers) == 0:
            centers = camera_centers_computed

        # Generate smooth path through camera centers
        val_centers = self._generate_video_path(centers, N_views)

        cameras = []
        len_poses = len(val_centers)
        times = [i / len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        # Compute average up vector from input cameras
        up_vectors = camera_Rs[:, :, 1]  # Y-axis is typically up
        avg_up = up_vectors.mean(axis=0)
        if np.linalg.norm(avg_up) < 1e-6:
            avg_up = np.array([0.0, 1.0, 0.0])
        avg_up = avg_up / np.linalg.norm(avg_up)

        # Load point cloud center for reference (but don't always look at it)
        ply_path = os.path.join(datadir, "points3D_multipleview.ply")
        if os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
            scene_center = np.asarray(pcd.points).mean(axis=0)
        else:
            scene_center = centers.mean(axis=0)

        # For each point on the path, find nearest input camera and interpolate orientation
        for idx, center in enumerate(val_centers):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]

            # Find the two nearest input cameras for interpolation
            dists = np.linalg.norm(centers - center, axis=1)
            nearest_idx = np.argmin(dists)

            # Get viewing direction from nearest camera
            nearest_R = camera_Rs[nearest_idx]
            forward = nearest_R[:, 2]  # Z-axis is forward in camera space

            # Slightly adjust toward scene center for better coverage
            # Mix 80% original direction + 20% toward scene center
            toward_center = scene_center - center
            if np.linalg.norm(toward_center) > 1e-6:
                toward_center = toward_center / np.linalg.norm(toward_center)
                forward = 0.8 * forward + 0.2 * toward_center

            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-6:
                forward = np.array([0.0, 0.0, 1.0])
            else:
                forward = forward / forward_norm

            # Construct orthonormal frame
            right = np.cross(forward, avg_up)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                # Fallback if forward is parallel to up
                fallback = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(forward, fallback)) > 0.99:
                    fallback = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, fallback)
                right_norm = np.linalg.norm(right)
            right = right / right_norm

            true_up = np.cross(right, forward)
            true_up = true_up / np.linalg.norm(true_up)

            # Construct rotation matrix: columns are [-right, up, forward]
            R_c2w = np.stack([-right, true_up, forward], axis=1)
            T = -R_c2w.transpose() @ center

            FovX = self.FovX
            FovY = self.FovY

            cameras.append(
                CameraInfo(
                    uid=idx,
                    R=R_c2w,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.shape[2],
                    height=image.shape[1],
                    time=time,
                    mask=None,
                )
            )
        return cameras

    def _generate_video_path(self, centers: np.ndarray, num_views: int) -> np.ndarray:
        """Generate a smooth interpolated path through the camera centers.

        Creates a path that smoothly interpolates between the input camera positions,
        staying close to the actual viewpoints rather than creating an orbit.
        """
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            return np.zeros((num_views, 3), dtype=np.float32)

        if len(centers) == 1:
            # If only one camera, return repeated positions
            return np.tile(centers[0], (num_views, 1))

        # Create smooth interpolation through camera centers using Catmull-Rom spline
        # Extend the path slightly beyond first/last cameras for smooth looping
        extended_centers = np.vstack([centers[-1:], centers, centers[:1]])

        # Generate parameter values for each camera
        # Use cumulative distance for better arc-length parameterization
        dists = np.sqrt(np.sum(np.diff(extended_centers, axis=0)**2, axis=1))
        cum_dists = np.concatenate([[0], np.cumsum(dists)])

        # Normalize to [0, 1] range
        cum_dists = cum_dists / cum_dists[-1]

        # Generate smooth interpolation parameters
        t_values = np.linspace(cum_dists[1], cum_dists[-2], num_views, endpoint=False)

        path = []
        for t in t_values:
            # Find the segment this t belongs to
            idx = np.searchsorted(cum_dists, t) - 1
            idx = np.clip(idx, 0, len(extended_centers) - 2)

            # Get the four control points for Catmull-Rom spline
            p0 = extended_centers[max(0, idx - 1)]
            p1 = extended_centers[idx]
            p2 = extended_centers[min(len(extended_centers) - 1, idx + 1)]
            p3 = extended_centers[min(len(extended_centers) - 1, idx + 2)]

            # Local parameter within segment
            local_t = (t - cum_dists[idx]) / max(cum_dists[idx + 1] - cum_dists[idx], 1e-6)
            local_t = np.clip(local_t, 0, 1)

            # Catmull-Rom interpolation
            t2 = local_t * local_t
            t3 = t2 * local_t

            point = 0.5 * (
                2 * p1 +
                (-p0 + p2) * local_t +
                (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                (-p0 + 3*p1 - 3*p2 + p3) * t3
            )
            path.append(point)

        return np.array(path, dtype=np.float32)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        # Allow PIL to load truncated files instead of erroring
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        path = self.image_paths[index]
        try:
            with Image.open(path) as im:
                img = self.transform(im)
        except (UnidentifiedImageError, OSError):
            # Fallback: create a black placeholder to keep the training loop running
            intr = None
            cam_id = self.camera_ids[index] if index < len(self.camera_ids) else None
            if cam_id is not None:
                intr = self.cam_intrinsics.get(cam_id, None)
            if intr is None:
                intr = next(iter(self.cam_intrinsics.values()))
            h, w = int(intr.height), int(intr.width)
            img = torch.zeros((3, h, w), dtype=torch.float32)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]
