import cv2
import numpy as np
import torch
import open3d as o3d
from config import Config
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader
from core.track import Tracker
from io_utils.dataset import dataset_factory
from io_utils.ground_truth import groundtruth_factory
from utilities.utils_draw import visualize_matches
from utilities.utils_depth import depth2pointcloud
from utilities.utils_metrics import estimate_error_R_T
import copy

def main():
    # Initialize config and data sources (same as test_track.py)
    config = Config()
    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings)
    
    # Load ground truth if available
    if groundtruth is not None:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        print("- Trajectory shape:", gt_traj3d.shape if gt_traj3d is not None else "None")
        print("- Poses shape:", gt_poses.shape if gt_poses is not None else "None")
    
    # Load two frames (using same frame indices as test_track.py)
    frame1 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 0, config)
    frame2 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 20, config)
    
    # Store ground truth poses
    gt_pose1 = frame1.gt_pose
    gt_pose2 = frame2.gt_pose
    print("Ground truth poses loaded")
    
    # Initialize tracker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = Tracker(camera_matrix=frame1.camera_matrix, device=device)
    
    # Track frame2 relative to frame1
    rel_pose, inliers = tracker.track_frames(frame1, frame2)
    if rel_pose is None:
        print("Tracking failed!")
        return
        
    print("\nEstimated relative pose:")
    print(rel_pose)
    print(f"\nFound {len(inliers)} inlier matches")
    
    # Create matches array for visualization
    valid_matches = torch.zeros((len(inliers), 2), dtype=torch.long)
    for i, idx in enumerate(inliers.flatten()):  # Flatten inliers array
        valid_matches[i, 0] = idx  # Index in first image
        valid_matches[i, 1] = idx  # Same index in second image
    
    # Visualize matches using inliers
    matched_image = visualize_matches(
        torch.from_numpy(frame1.image).permute(2,0,1).float()/255.0,
        torch.from_numpy(frame2.image).permute(2,0,1).float()/255.0,
        torch.from_numpy(frame1.keypoints),
        torch.from_numpy(frame2.keypoints),
        valid_matches
    )
    cv2.imshow("Feature Matches (Inliers)", matched_image)
    cv2.waitKey(1)
    
    # Create point clouds for visualization
    fx = frame1.camera_matrix[0,0]
    fy = frame1.camera_matrix[1,1]
    cx = frame1.camera_matrix[0,2]
    cy = frame1.camera_matrix[1,2]
    
    # Create and colorize point clouds (blue for frame1, green tint for frame2)
    pcd1 = depth2pointcloud(frame1.depth, frame1.image, fx, fy, cx, cy, max_depth=10000.0, min_depth=0.0)
    pcd2 = depth2pointcloud(frame2.depth, frame2.image, fx, fy, cx, cy, max_depth=10000.0, min_depth=0.0)
    
    # Create Open3D point clouds
    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1.points)
    pcd1_o3d.colors = o3d.utility.Vector3dVector(pcd1.colors)
    # Add blue tint
    colors1 = np.asarray(pcd1_o3d.colors)
    colors1[:, 2] *= 0.5  # Reduce blue channel
    pcd1_o3d.colors = o3d.utility.Vector3dVector(colors1)
    
    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2.points)
    pcd2_o3d.colors = o3d.utility.Vector3dVector(pcd2.colors)
    # Add green tint
    colors2 = np.asarray(pcd2_o3d.colors)
    colors2[:, 0] *= 0.5  # Reduce red channel
    pcd2_o3d.colors = o3d.utility.Vector3dVector(colors2)
    
    # Create coordinate frames
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    pose1_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    pose2_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    # Transform using estimated pose
    pcd2_o3d.transform(np.linalg.inv(rel_pose))
    pose2_frame.transform(rel_pose)
    
    # Create ground truth coordinate frame for comparison
    pose_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    T_rel_gt = np.linalg.inv(gt_pose2) @ gt_pose1
    pose_gt.transform(T_rel_gt)
    pose_gt.paint_uniform_color([0, 1, 0])  # Green for ground truth
    
    print("\nEstimated vs Ground Truth:")
    print("Estimated relative pose:")
    print(rel_pose)
    print("\nGround truth relative pose:")
    print(T_rel_gt)

    # Estimate error in rotation and translation
    angle_err, t_err = estimate_error_R_T(rel_pose, T_rel_gt)
    print(f"\nRotation error: {angle_err:.2f} degrees")
    print(f"Translation error: {t_err:.2f} meters")
    
    # Visualize point clouds and poses
    o3d.visualization.draw_geometries([
        pcd1_o3d,          # Frame 1 point cloud
        pcd2_o3d,          # Frame 2 point cloud (transformed)
        origin,            # World origin
        pose1_frame,       # Frame 1 pose
        pose2_frame,       # Estimated Frame 2 pose
        pose_gt            # Ground truth Frame 2 pose
    ])

    pcd1_core_points, pcd1_core_colors = frame1.get_points(global_coords=True)
    pcd2_core_points, pcd2_core_colors = frame2.get_points(global_coords=True)
    pcd1_core = o3d.geometry.PointCloud()
    pcd1_core.points = o3d.utility.Vector3dVector(pcd1_core_points)
    pcd1_core.colors = o3d.utility.Vector3dVector(pcd1_core_colors)
    pcd2_core = o3d.geometry.PointCloud()
    pcd2_core.points = o3d.utility.Vector3dVector(pcd2_core_points)
    pcd2_core.colors = o3d.utility.Vector3dVector(pcd2_core_colors)
    o3d.visualization.draw_geometries([pcd1_core, pcd2_core, origin])


if __name__ == "__main__":
    main()
