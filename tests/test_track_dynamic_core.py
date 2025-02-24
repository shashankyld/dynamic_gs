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

def create_dynamic_mask(img_shape, center, radius):
    """Create a sample dynamic mask with a circle"""
    mask = np.ones(img_shape[:2], dtype=bool)  # True = static, False = dynamic
    y, x = np.ogrid[:img_shape[0], :img_shape[1]]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask[dist_from_center <= radius] = False
    return mask

def main():
    # Initialize config and data sources
    config = Config()
    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings)
    
    # Load ground truth if available
    if groundtruth is not None:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        print("- Trajectory shape:", gt_traj3d.shape if gt_traj3d is not None else "None")
        print("- Poses shape:", gt_poses.shape if gt_poses is not None else "None")
    
    # Load two frames
    frame1 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 0, config)
    frame2 = get_frame_from_pyslam_dataloader(dataset, groundtruth, 20, config)
    
    # Create a sample dynamic mask (simulating a moving object in the center)
    h, w = frame1.image.shape[:2]
    dynamic_mask = create_dynamic_mask(frame1.image.shape, center=(w//2, h//2), radius=200)
    
    # Visualize the mask
    masked_img = frame1.image.copy()
    masked_img[~dynamic_mask] = masked_img[~dynamic_mask] * 0.3  # Darken dynamic regions
    cv2.imshow("Dynamic Mask", masked_img)
    cv2.waitKey(100)
    
    # Initialize tracker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = Tracker(camera_matrix=frame1.camera_matrix, device=device)
    
    # Track with and without mask for comparison
    print("\nTracking without mask:")
    rel_pose_no_mask, inliers_no_mask = tracker.track_frames(frame1, frame2)
    
    print("\nTracking with mask:")
    rel_pose_masked, inliers_masked = tracker.track_frames_with_mask(frame1, frame2, dynamic_mask)
    
    if rel_pose_masked is None or rel_pose_no_mask is None:
        print("Tracking failed!")
        return
    
    print(f"Number of inliers without mask: {len(inliers_no_mask)}")
    print(f"Number of inliers with mask: {len(inliers_masked)}")
    
    # Compute ground truth relative pose
    gt_pose1 = frame1.gt_pose
    gt_pose2 = frame2.gt_pose
    T_rel_gt = np.linalg.inv(gt_pose2) @ gt_pose1
    
    # Compare errors
    angle_err_no_mask, t_err_no_mask = estimate_error_R_T(rel_pose_no_mask, T_rel_gt)
    angle_err_masked, t_err_masked = estimate_error_R_T(rel_pose_masked, T_rel_gt)
    
    print("\nError metrics:")
    print(f"Without mask - Rotation error: {angle_err_no_mask:.2f}°, Translation error: {t_err_no_mask:.3f}m")
    print(f"With mask    - Rotation error: {angle_err_masked:.2f}°, Translation error: {t_err_masked:.3f}m")
    
    # Visualize results
    # Create point clouds
    pcd1 = depth2pointcloud(frame1.depth, frame1.image, 
                           frame1.fx, frame1.fy, frame1.cx, frame1.cy,
                           max_depth=10000.0, min_depth=0.0)
    pcd2 = depth2pointcloud(frame2.depth, frame2.image,
                           frame2.fx, frame2.fy, frame2.cx, frame2.cy,
                           max_depth=10000.0, min_depth=0.0)
    
    # Create visualization elements
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    
    # Create point clouds for masked tracking result
    pcd1_masked = o3d.geometry.PointCloud()
    pcd1_masked.points = o3d.utility.Vector3dVector(pcd1.points)
    pcd1_masked.colors = o3d.utility.Vector3dVector(pcd1.colors)
    pcd1_masked.paint_uniform_color([1, 0, 0])  # Red
    
    pcd2_masked = o3d.geometry.PointCloud()
    pcd2_masked.points = o3d.utility.Vector3dVector(pcd2.points)
    pcd2_masked.colors = o3d.utility.Vector3dVector(pcd2.colors)
    pcd2_masked.paint_uniform_color([0, 0, 1])  # Blue
    
    # Transform using masked pose estimate
    pcd2_masked.transform(np.linalg.inv(rel_pose_masked))
    
    # Create coordinate frames
    pose_masked = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    pose_masked.transform(rel_pose_masked)
    
    pose_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    pose_gt.transform(T_rel_gt)
    pose_gt.paint_uniform_color([0, 1, 0])  # Green for ground truth
    
    # Visualize
    print("\nVisualization Legend:")
    print("Red: Frame 1 point cloud")
    print("Blue: Frame 2 point cloud (transformed by masked tracking)")
    print("Green: Ground truth pose")
    print("Regular axes: Estimated pose with mask")
    
    o3d.visualization.draw_geometries([
        pcd1_masked,  # Frame 1
        pcd2_masked,  # Frame 2 (transformed)
        origin,       # World origin
        pose_masked,  # Estimated pose
        pose_gt       # Ground truth pose
    ])


    

if __name__ == "__main__":
    main()
