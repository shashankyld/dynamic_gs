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
from utilities.utils_metrics import estimate_error_R_T
from io_utils.dataset import dataset_factory
from io_utils.ground_truth import groundtruth_factory
import copy
import tqdm



config = Config()
groundtruth = groundtruth_factory(config.dataset_settings)
dataset = dataset_factory(config)
depthmapfactor = config.cam_settings["DepthMapFactor"]
print("depthmapfactor: ", depthmapfactor)
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


# ### Visualize the trajectory with axes in Open3D
# traj = []
# for i in range(gt_poses.shape[0]+1):
#     if i == 0:
#         origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#         traj.append(origin_axis)
#     else:
#         pose = gt_poses[i-1]
#         axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
#         axis = axis.transform(pose)
#         traj.append(axis)
# o3d.visualization.draw_geometries(traj)



frame1 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 0, config)
frame2 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 20, config)

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
# CHANGE COLORS - BLUE TINT
pcd1_colors = np.array(pcd1.colors)
pcd1_colors[:, 2] *= 0.5
pcd1.colors = o3d.utility.Vector3dVector(pcd1_colors)
pcd1_copy = copy.deepcopy(pcd1)
# Transform the point cloud to the world frame 
pcd1.transform(gt_pose1)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pc2.points)
pcd2.colors = o3d.utility.Vector3dVector(pc2_colors)
# CHANGE COLORS - GREEN TINT
pcd2_colors = np.array(pcd2.colors)
pcd2_colors[:, 0] *= 0.5 
pcd2.colors = o3d.utility.Vector3dVector(pcd2_colors)
pcd2_copy = copy.deepcopy(pcd2)
# Transform the point cloud to the world frame 
pcd2.transform(gt_pose2)


# Origin 
origin_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
pose1_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
pose1_axis = pose1_axis.transform(gt_pose1)
pose2_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
pose2_axis = pose2_axis.transform(gt_pose2)

# Visualize point clouds
print("Visualizing point clouds - currently disabled")
o3d.visualization.draw_geometries([   pcd1, pose1_axis, pcd2, pose2_axis])



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
# Explicitly set keypoints into frame objects (convert to numpy)
frame1.keypoints = kpts0.cpu().numpy()
frame2.keypoints = kpts1.cpu().numpy()

# Visualize matches
matched_image = visualize_matches(img1_device, img2_device, kpts0, kpts1, matches)
cv2.imshow("Matched Image", matched_image)
cv2.waitKey(1000)



# Impliment PnP - RANSAC tracking with OpenCV only with the matched keypoints

# 1. Extract the matched keypoints
# 2. Match the keypoints
# 3. Implement PnP RANSAC
# 4. Get the transformation matrix
# 5. Transform the point cloud
# 6. Visualize the point cloud

# Prepare correspondences for PnP: using frame1 to generate 3D points and frame2 for their 2D locations.
obj_points = []  # 3D points (in frame1 coordinates)
img_points = []  # 2D points in frame2 image

# Camera intrinsics from frame1
camera_matrix = frame1.camera_matrix.astype(np.float64)
fx, fy = camera_matrix[0,0], camera_matrix[1,1]
cx, cy = camera_matrix[0,2], camera_matrix[1,2]

# For each match, use the keypoint from frame1 and get depth from frame1's depth image.
for m in matches:
    print("m: ", m)
    idx1, idx2 = int(m[0]), int(m[1])
    pt1 = frame1.keypoints[idx1]  # [x, y] in frame1
    pt2 = frame2.keypoints[idx2]  # [x, y] in frame2
    
    # Round coordinates and check bounds
    u, v = int(round(pt1[0])), int(round(pt1[1]))
    if u < 0 or v < 0 or v >= frame1._depth.shape[0] or u >= frame1._depth.shape[1]:
        continue
    depth_value = frame1._depth[v, u]
    if depth_value <= 0:
        continue
    # Backproject pixel (u, v) to 3D point using pinhole model
    X = (pt1[0] - cx) * depth_value / fx
    Y = (pt1[1] - cy) * depth_value / fy
    Z = depth_value
    obj_points.append([X, Y, Z])
    img_points.append(pt2)  # use pt2 from frame2

obj_points = np.array(obj_points, dtype=np.float64)
img_points = np.array(img_points, dtype=np.float64)
print(f"Using {obj_points.shape[0]} correspondences for PnP.")

if obj_points.shape[0] < 6:
    print("Not enough correspondences for PnP. Exiting.")
    exit(1)

# Run PnP RANSAC (using OpenCV)
retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points, img_points, camera_matrix, None,
                                                   flags=cv2.SOLVEPNP_ITERATIVE,
                                                   reprojectionError=8.0,
                                                   iterationsCount=100,
                                                   confidence=0.99)
if not retval:
    print("PnP RANSAC did not succeed.")
    exit(1)
print(f"Found {len(inliers)} inliers out of {obj_points.shape[0]} correspondences.")

# Convert rotation vector to rotation matrix
R, _ = cv2.Rodrigues(rvec)
# Build the 4x4 transformation matrix: pose of frame2 relative to frame1
T_relative = np.eye(4)
T_relative[:3, :3] = R
T_relative[:3, 3] = tvec.flatten()

print("Estimated relative pose (frame1 -> frame2):")
print(T_relative)

# Visualize the result: transform frame1's point cloud and show over frame2's
pcd1 = depth2pointcloud(frame1._depth, frame1._image, fx, fy, cx, cy, max_depth=10000.0, min_depth=0.0)
pcd2 = depth2pointcloud(frame2._depth, frame2._image, fx, fy, cx, cy, max_depth=10000.0, min_depth=0.0)
# Transform pcd1 using T_relative
ones = np.ones((pcd1.points.shape[0], 1))
pcd1_hom = np.hstack((pcd1.points, ones))
pcd1_transformed = (T_relative @ pcd1_hom.T).T[:, :3]

# Create Open3D point clouds for visualization
pcd1_o3d = o3d.geometry.PointCloud()
pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1_transformed)
pcd1_o3d.colors = o3d.utility.Vector3dVector(pcd1.colors)
# Add Blue tint to the transformed point cloud
pcd1_o3d_colors = np.array(pcd1_o3d.colors)
pcd1_o3d_colors[:, 2] *= 0.5
pcd1_o3d.colors = o3d.utility.Vector3dVector(pcd1_o3d_colors)
pcd2_o3d = o3d.geometry.PointCloud()
pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2.points)
# Color frame2 cloud in blue
pcd2_o3d.colors = o3d.utility.Vector3dVector(pcd2.colors)
# Add Green tint to the frame2 point cloud
pcd2_o3d_colors = np.array(pcd2_o3d.colors)
pcd2_o3d_colors[:, 0] *= 0.5
pcd2_o3d.colors = o3d.utility.Vector3dVector(pcd2_o3d_colors)



# Visualize the point clouds and poses computed by the tracker
origin_tracking = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
pose1_tracking = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
pose2_tracking = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
# Transform the coordinate frames using T_relative
pose1_tracking = pose1_tracking.transform(T_relative)

T_rel_gt = np.linalg.inv(gt_pose2) @ gt_pose1


print("T_rel_gt: ", T_rel_gt)

# ERROR IN RELATIVE POSE CALCULATION
angle_err, t_err = estimate_error_R_T(T_relative, T_rel_gt)
print("Rotation error (degrees):", angle_err)
print("Translation error (meters):", t_err)




pose1_tracking_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
pose1_tracking_gt = pose1_tracking_gt.transform(T_rel_gt)



# Color as green
pose1_tracking_gt.paint_uniform_color([0, 1, 0])








print("Displaying point clouds: frame1 transformed (red) and frame2 (blue)")

o3d.visualization.draw_geometries([pcd1_o3d, pcd2_o3d, pose1_tracking, pose2_tracking,  origin_tracking,pose1_tracking_gt])



