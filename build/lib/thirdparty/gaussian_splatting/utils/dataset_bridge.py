from thirdparty.gaussian_splatting.utils.camera_utils import Camera
from thirdparty.gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from config import Config
from io_utils.dataset import dataset_factory
from io_utils.ground_truth import groundtruth_factory
from pyquaternion import Quaternion
import numpy as np
import torch


def get_camera_info_from_pyslam_dataloader_insteadofgs(dataset, ground_truth,img_id, config):
    """
    Converts a PYSLAM dataset object to a CameraInfo object 
    compatible with the Gaussian Splatting pipeline.

    Args:
        dataset: The PYSLAM dataset object from dataset_factory.
        img_id: The index of the image/frame in the dataset.

    Returns:
        A CameraInfo object.
    """

    # 1. Get image dimensions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt_img = dataset.getImage(img_id)
    gt_img = (
            torch.from_numpy(gt_img / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=device, dtype=torch.float32)
        ) # This is 3xHxW
    # Change to HxWx3 and keep it on device and same dtype
    # gt_img = gt_img.permute(1, 2, 0)
    print("Shape of the gt_image according to camera_info", gt_img.shape)
    gt_depth = dataset.getDepth(img_id)
    print("Shape of the gt_depth according to camera_info", gt_depth.shape)
    height, width = gt_img.shape[1], gt_img.shape[2]
    gt_timestamp, x, y, z, qx, qy, qz, qw, abs_scale = ground_truth.getTimestampPoseAndAbsoluteScale(img_id)
    gt_pose = np.eye(4)
    gt_pose[:3, :3] = Quaternion(qw, qx, qy, qz).rotation_matrix
    gt_pose[:3, 3] = np.array([x, y, z])
    fx = config.cam_settings['Camera.fx']
    fy = config.cam_settings['Camera.fy']
    cx = config.cam_settings['Camera.cx']
    cy = config.cam_settings['Camera.cy']
    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))
    projection_matrix = getProjectionMatrix2(
                znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=width, H=height
            ).transpose(0, 1)    # Create a CameraInfo object
    camera_info = Camera(img_id, gt_img, gt_depth, gt_pose, projection_matrix, fx, fy, cx, cy, fovx, fovy, height, width, device=device)
    return camera_info
    
    
    
## TESTING 

config = Config()
ground_truth = groundtruth_factory(config.dataset_settings)
dataset = dataset_factory(config)
camera_info = get_camera_info_from_pyslam_dataloader_insteadofgs(dataset, ground_truth, img_id=0, config=config)
print(camera_info)

# Now you can use camera_info in the Gaussian Splatting pipeline