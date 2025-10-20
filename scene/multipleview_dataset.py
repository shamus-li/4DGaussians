import os
import numpy as np
from typing import Dict, List
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat
from scene.dataset_readers import CameraInfo, fetchPly
from scene.neural_3D_dataset_NDC import get_spiral
from torchvision import transforms as T


class multipleview_dataset(Dataset):
    def __init__(
        self,
        cam_extrinsics,
        cam_intrinsics,
        cam_folder,
        split
    ):
        self.focal = [cam_intrinsics[1].params[0], cam_intrinsics[1].params[0]]
        height=cam_intrinsics[1].height
        width=cam_intrinsics[1].width
        self.FovY = focal2fov(self.focal[0], height)
        self.FovX = focal2fov(self.focal[0], width)
        self.transform = T.ToTensor()
        self.image_paths, self.image_poses, self.image_times= self.load_images_path(cam_folder, cam_extrinsics,cam_intrinsics,split)
        if split=="test":
            self.video_cam_infos=self.get_video_cam_infos(cam_folder)
        
    
    def load_images_path(self, cam_folder, cam_extrinsics,cam_intrinsics,split):
        image_paths=[]
        image_poses=[]
        image_times=[]
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

        return image_paths, image_poses,image_times
    
    def get_video_cam_infos(self,datadir):
        poses_arr = np.load(os.path.join(datadir, "poses_bounds_multipleview.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        near_fars = poses_arr[:, -2:]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        N_views = 300
        val_poses = get_spiral(poses, near_fars, N_views=N_views)

        cameras = []
        len_poses = len(val_poses)
        times = [i / len_poses for i in range(len_poses)]
        image = Image.open(self.image_paths[0])
        image = self.transform(image)

        ply_path = os.path.join(datadir, "points3D_multipleview.ply")
        if os.path.exists(ply_path):
            pcd = fetchPly(ply_path)
            target = np.asarray(pcd.points).mean(axis=0)
        else:
            target = np.zeros(3, dtype=np.float32)
        self.mean_center = target
        up_hint = val_poses[:, :3, 1].mean(axis=0)
        if np.linalg.norm(up_hint) < 1e-6:
            up_hint = np.array([0.0, 1.0, 0.0])

        for idx, p in enumerate(val_poses):
            image_path = None
            image_name = f"{idx}"
            time = times[idx]

            pose = np.eye(4)
            pose[:3, :4] = p[:3, :4]

            C = pose[:3, 3]

            forward = target - C
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-6:
                forward = pose[:3, 2]
                forward_norm = np.linalg.norm(forward)
            forward = forward / forward_norm

            right = np.cross(forward, up_hint)
            right_norm = np.linalg.norm(right)
            if right_norm < 1e-6:
                # fall back to an orthogonal axis if the hint is collinear
                fallback = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(forward, fallback)) > 0.99:
                    fallback = np.array([0.0, 1.0, 0.0])
                right = np.cross(forward, fallback)
                right_norm = np.linalg.norm(right)
            right = right / right_norm

            true_up = np.cross(right, forward)
            true_up = true_up / np.linalg.norm(true_up)

            R_c2w = np.stack([-right, true_up, forward], axis=1)
            T = -R_c2w.transpose() @ C

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
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        img = self.transform(img)
        return img, self.image_poses[index], self.image_times[index]
    def load_pose(self,index):
        return self.image_poses[index]
