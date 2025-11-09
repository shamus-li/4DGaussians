from torch.utils.data import Dataset
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov
import torch
from utils.camera_utils import loadCam
from utils.graphics_utils import focal2fov
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
    def __getitem__(self, index):
        # breakpoint()

        if self.dataset_type != "PanopticSports":
            try:
                sample = self.dataset[index]
                if isinstance(sample, tuple) and len(sample) == 3:
                    image, w2c, time = sample
                    R, T = w2c
                    mask = None
                    if torch.is_tensor(image) and image.dim() == 3 and image.shape[0] > 3:
                        mask = image[3:4, ...].clone()
                        image = image[:3, ...]
                    # Prefer per-image intrinsics if available on the dataset
                    if hasattr(self.dataset, "cam_intrinsics") and hasattr(self.dataset, "camera_ids"):
                        cam_id = self.dataset.camera_ids[index]
                        intr = self.dataset.cam_intrinsics.get(cam_id, None)
                        if intr is not None:
                            fx = float(intr.params[0])
                            fy = float(intr.params[1]) if intr.model in ["PINHOLE", "OPENCV"] and len(intr.params) > 1 else fx
                            FovX = focal2fov(fx, intr.width)
                            FovY = focal2fov(fy, intr.height)
                        else:
                            FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                            FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                    else:
                        FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                        FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                else:
                    # CameraInfo-like object
                    caminfo = sample
                    image = caminfo.image
                    R = caminfo.R
                    T = caminfo.T
                    FovX = caminfo.FovX
                    FovY = caminfo.FovY
                    time = caminfo.time
                    mask = caminfo.mask
            except IndexError:
                # Propagate IndexError so Python's implicit sequence iterator knows to stop.
                raise
            except Exception as e:
                # As a last resort, propagate a clearer error with index context
                raise RuntimeError(f"Dataset item {index} could not be parsed: {e}")
            return Camera(colmap_id=index,R=R,T=T,FoVx=FovX,FoVy=FovY,image=image,gt_alpha_mask=mask,
                              image_name=f"{index}",uid=index,data_device=torch.device("cuda"),time=time,
                              mask=mask)
        else:
            return self.dataset[index]
    def __len__(self):
        
        return len(self.dataset)
