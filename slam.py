#!/usr/bin/env python3
from config import Config
from io_utils.dataset import dataset_factory, SensorType
from io_utils.ground_truth import groundtruth_factory
import logging 
from utilities.utils_depth import depth2pointcloud, depth2pcd
import cv2
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint
import torch
import numpy as np
import open3d as o3d
from utilities.utils_draw import draw_torch_image

if __name__ == "__main__":
    # SETTING UP DATASET PARAMS
    config = Config()
    dataset = dataset_factory(config) 
    depthmapfactor = config.cam_settings["DepthMapFactor"]
    depth_scale = 1/depthmapfactor

    # GT DATA
    groundtruth = groundtruth_factory(config.dataset_settings)
    
    # PIPELINE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SETTING UP SLAM PARAMETERS
    num_features = config.num_features_to_extract if config.num_features_to_extract > 0 else 5000
    kNumFramesAway = config.NumFramesAway
    
    # Feature extraction and matching setup
    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Load groundtruth trajectory if available
    if groundtruth is not None:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        if gt_poses is None or len(gt_poses) == 0:
            print("Error: No ground truth poses loaded!")
            exit(1)

    """
    USAGE EXAMPLES:
        gt_t1 = groundtruth.getTimestampPoseMatrix(0)
        print(gt_t1)
        # END the script here
        exit(0)
    """
    # Initialize processing variables
    starting_img_id = config.start_frame_id
    ending_img_id = config.end_frame_id
    img_id = starting_img_id
    global_poses = {}
    accumulated_clouds = {}


    while True: 
        # Get current frame data
        if dataset.isOk(): 
            img = dataset.getImage(img_id)
            depth_img = dataset.getDepth(img_id) * depth_scale
            
            if img is not None:
                timestamp = dataset.getTimestamp()
                point_cloud = depth2pcd(
                    depth_img, img,
                    config.cam_settings["Camera.fx"], 
                    config.cam_settings["Camera.fy"],
                    config.cam_settings["Camera.cx"], 
                    config.cam_settings["Camera.cy"],
                    max_depth=100000.0, min_depth=0.0
                )
                
                # Convert to torch tensor
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                
                # Get pose from groundtruth if available
                timestamp, global_poses[img_id] = groundtruth.getTimestampPoseMatrix(img_id)

                # Transform point cloud to global coordinates 
                point_cloud.transform(global_poses[img_id])
                # Store point cloud if valid

                accumulated_clouds[img_id] = point_cloud
                
                
                
                
                # Visualize current frame
                draw_torch_image(img_tensor)
                cv2.waitKey(1)

        # Process next frame
        img_id += 1
        
        if img_id > ending_img_id:
            print("Reached end frame ID")
            # Visualize accumulated point clouds
            pcd_list = []
            poses_o3d = []
            
            for i in range(len(accumulated_clouds)):
                #downsample
                accumulated_clouds[i].voxel_down_sample(voxel_size=0.05)
                pcd_list.append(accumulated_clouds[i])
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                axes.transform(global_poses[i])
                poses_o3d.append(axes)

            o3d.visualization.draw_geometries(pcd_list + poses_o3d)
            break