import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from config import Config
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader
from core.frame import Frame
from core.track import Tracker
from thirdparty.LightGlue.lightglue import SuperPoint, LightGlue
from thirdparty.LightGlue.lightglue.utils import rbd
import open3d as o3d
from utilities.utils_depth import depth2pointcloud
from utilities.utils_draw import visualize_matches  
from io_utils.dataset import dataset_factory
from io_utils.ground_truth import groundtruth_factory


config = Config()
groundtruth = groundtruth_factory(config.dataset_settings)
dataset = dataset_factory(config)
depthmapfactor = config.cam_settings["DepthMapFactor"]
depth_scale = 1/depthmapfactor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

extractor = SuperPoint(max_num_keypoints=5000).eval().to(device) 
matcher = LightGlue(features="superpoint").eval().to(device)

# Loading Groundtruth trajectory
if groundtruth is not None:
    # Initialize with 6D trajectory instead of 3D
    gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
    print("Loaded ground truth:")
    print("- Trajectory shape:", gt_traj3d.shape if gt_traj3d is not None else "None")
    print("- Poses shape:", gt_poses.shape if gt_poses is not None else "None")
    print("- Timestamps range:", gt_timestamps[0], "to", gt_timestamps[-1] if gt_timestamps is not None else "None")
    
    # Verify poses are properly loaded
    if gt_poses is None or len(gt_poses) == 0:
        print("Error: No ground truth poses loaded!")
        exit(1)

print("gt_traj3d: ", gt_traj3d.shape)

frame1 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 0, config)
frame2 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 60, config)

img1 = frame1._image 
img2 = frame2._image
depth1 = frame1._depth
depth2 = frame2._depth

gt_pose1 = frame1.gt_pose 
gt_pose2 = frame2.gt_pose
print("gt_pose1: ", gt_pose1)
print("gt_pose2: ", gt_pose2)

c_fx = frame1.fx
c_fy = frame1.fy
c_cx = frame1.cx
c_cy = frame1.cy

print("c_fx: ", c_fx)
print("c_fy: ", c_fy)
print("c_cx: ", c_cx)
print("c_cy: ", c_cy)

pc1 = depth2pointcloud(depth1, img1, c_fx, c_fy, c_cx, c_cy, max_depth=100000.0, min_depth=0.0)
pc1_colors = pc1.colors
pc2 = depth2pointcloud(depth2, img2, c_fx, c_fy, c_cx, c_cy, max_depth=100000.0, min_depth=0.0)
pc2_colors = pc2.colors
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pc1.points)
pcd1.colors = o3d.utility.Vector3dVector(pc1_colors)
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pc2.points)
pcd2.colors = o3d.utility.Vector3dVector(pc2_colors)


img1_device = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
depth1_device = torch.from_numpy(depth1.astype(np.float32)).unsqueeze(0)
img2_device = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
depth2_device = torch.from_numpy(depth2.astype(np.float32)).unsqueeze(0)

# Feature extraction and matching
img1_feat = extractor.extract(img1_device.to(device))
img2_feat = extractor.extract(img2_device.to(device))
matches01 = matcher({'image0': img1_feat, 'image1': img2_feat})
feats0, feats1, matches01 = [
                rbd(x) for x in [img1_feat, img2_feat, matches01]
            ]  # remove batch dimension

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[:, 0]], kpts1[matches[:, 1]]

# Visualize matches
matched_image = visualize_matches(img1_device, img2_device, kpts0, kpts1, matches)
# cv2.imshow("Matched Image", matched_image)
# cv2.waitKey(0)

origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
camera_axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
camera_axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
# Visualize point clouds 
print("pcd1: ", pcd1)
print("pcd2: ", pcd2)
# o3d.visualization.draw_geometries([pcd1, pcd2, camera_axis1, camera_axis2])


# Check gt-pose quality by transforming the pcd1 with gt_pose1 and pcd2 with gt_pose2
print("gt_pose1: ", gt_pose1)
print("gt_pose2: ", gt_pose2)
# Apply inverse transformation
pcd1_gt = pcd1.transform(gt_pose1)
pcd2_gt = pcd2.transform(gt_pose2)
camera_axis1_gt = camera_axis1.transform(gt_pose1)
camera_axis2_gt = camera_axis2.transform(gt_pose2)
o3d.visualization.draw_geometries([pcd1_gt, pcd2_gt, camera_axis1_gt, camera_axis2_gt, origin_axis])

# Impliment PnP - RANSAC tracking with OpenCV only with the matched keypoints
# Define the camera intrinsic matrix
