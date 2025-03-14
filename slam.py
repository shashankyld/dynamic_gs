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
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader
from utilities.utils_edges import (find_matching_edges, find_dynamic_edges, 
                                 find_connected_components, create_dynamic_mask,
                                 visualize_edges, visualize_dynamic_components, EdgeTracker)
from core.slam_system import SLAMSystem as SLAM
from utilities.utils_metrics import estimate_error_R_T
from utilities.utils_delaunay import *
from core.trajectory_aux import Trajectory
from core.pc_map_aux import PC_Map


if __name__ == "__main__":
    # SETTING UP DATASET PARAMS
    config = Config()
    dataset = dataset_factory(config) 
    depthmapfactor = config.cam_settings["DepthMapFactor"]
    depth_scale = 1/depthmapfactor

    # Camera matrix (intrinsics)
    camera_matrix = np.array([
        [config.cam_settings["Camera.fx"], 0, config.cam_settings["Camera.cx"]],
        [0, config.cam_settings["Camera.fy"], config.cam_settings["Camera.cy"]],
        [0, 0, 1]
    ], dtype=np.float64)

    # GT DATA
    groundtruth = groundtruth_factory(config.dataset_settings)
    
    # PIPELINE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SETTING UP SLAM PARAMETERS
    num_features = config.num_features_to_extract if config.num_features_to_extract > 0 else 5000
    kNumFramesAway = config.NumFramesAway
    kNumLocalKFs = config.NUM_LOCAL_KEYFRAMES

    # Feature extraction and matching setup
    slam = SLAM(camera_matrix, num_features, num_local_keyframes=kNumLocalKFs, device=device)

    ## SETTING CONFIG TO SLAM OBJECT
    slam.dataset = dataset  
    slam.groundtruth = groundtruth
    slam.config = config

    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)


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
    accumulated_pc = PC_Map()
    global_traj = Trajectory()
    local_traj = Trajectory()
    


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
                print("Getting ground truth pose for frame: ", img_id)
                timestamp, global_traj.trajectory[img_id] = groundtruth.getTimestampPoseMatrix(img_id, set_id_to_eye4=starting_img_id)

                
                
                
                
                # Visualize current frame
                draw_torch_image(img_tensor)
                cv2.waitKey(1)

                if img_id == starting_img_id: 
                    print("Initializing SLAM")
                    print("Processing frame: ", img_id)
                    curr_frame = get_frame_from_pyslam_dataloader(dataset, groundtruth, img_id, config)
                    slam.initialize(curr_frame)
                    print("Initialized SLAM")
                    print("SLAM: ", slam)
                    print("Tracker: ", slam.tracker)
                    print("Map: ", slam.map)
                    # Print Current Frame 
                    print("Current Frame: ", curr_frame)
                    print("Current Frame Pose: ", curr_frame.pose)
                    print("GT Pose: ", global_traj.trajectory[img_id])

                    local_traj.trajectory[img_id] = curr_frame.pose


                    ## CHECK IF DELAUNAY TRIANGULATION SHOULD BE CREATED 
                    # For first frame, yes 
                    Delaunay_G = delaunay_triangulation(curr_frame)
                    Delaunay_img = draw_delaunay_triangulation(Delaunay_G, curr_frame)
                    
                    # Store Delaunay triangulation in frame
                    curr_frame.delaunay = Delaunay_G

                    prev_delaunay_id = starting_img_id


                    prev_frame = curr_frame

                else:
                    print("Processing frame: ", img_id)
                    curr_frame = get_frame_from_pyslam_dataloader(dataset, groundtruth, img_id, config)
                    
                    dynamic_mask = None
                    curr_frame._dynamic_mask = dynamic_mask

                    # Track features
                    print("Tracking frame, ", img_id)
                    tracking_result = slam.track_frame(curr_frame)
                    print("Tracking Result: ", tracking_result)
                    print("Current Frame: ", curr_frame)
                    print("Current Frame Pose: ", curr_frame.pose)
                    print("GT Pose: ", global_traj.trajectory[img_id])

                    local_traj.trajectory[img_id] = curr_frame.pose



                    # Find the neared frame KF that has delaunay, use its delaunay edges for comparisons.
                    prev_delaunay_id = slam.map.local_keyframes[-1]
                    print("Current Frame ID: ", img_id)
                    print("Prev Delaunay ID: ", prev_delaunay_id)

                    if img_id - starting_img_id > kNumFramesAway:
                        # Compare delaunay between frames
                        matches_curr_prev, matches_curr_k, matches_prev_k, common_matches = \
                            matches_with_k_frames_away_with_prev_delaunay_edges(curr_frame, slam, kNumFramesAway)
                            
                        # G_all_frames(curr_frame=curr_frame, slam=slam)
                        # get_static_dynamic_edges(curr_frame, slam)
                        if config.DYNAMIC_PROCESSING_METHOD == "batch":
                            G_static, G_dynamic= get_static_dynamic_edges_batch(curr_frame, slam)
                        elif config.DYNAMIC_PROCESSING_METHOD == "k_frames_away":
                            G_static, G_dynamic = get_static_dynamic_edges(curr_frame, slam) 
                        
                        

                    prev_frame = curr_frame 


                # Print Relative Pose Error - curr_frame vs prev_frame (tracked vs ground truth) and curr_frame (tracked) vs ground truth
                tracked_odometry = np.linalg.inv(prev_frame.pose) @ curr_frame.pose
                gt_odometry = np.linalg.inv(global_traj.trajectory[prev_frame.id]) @ global_traj.trajectory[curr_frame.id]
                print("Relative Odometry Error deg/m: ", estimate_error_R_T(tracked_odometry, gt_odometry))
                print("Relative Pose Error deg/m: ", estimate_error_R_T(curr_frame.pose, global_traj.trajectory[curr_frame.id]))
                
                # Transform point cloud to global coordinates 
                point_cloud.transform(global_traj.trajectory[img_id])
                # point_cloud.transform(local_traj.trajectory[img_id])
                
                # Store point cloud 
                accumulated_pc.add(img_id, point_cloud)
                


        # Process next frame
        img_id += 1


        if img_id > ending_img_id:
            # exit(0)

            # Visualize GT poses and tracked poses
            poses_o3d = []
            for i in range(starting_img_id, ending_img_id + 1):
                if i % 10 != 0: continue
                else:
                    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                    axes.transform(global_traj.trajectory[i])
                    # Color GT poses in green
                    axes.paint_uniform_color([0, 1, 0])
                    poses_o3d.append(axes)

                    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                    axes.transform(local_traj.trajectory[i])
                    # Color tracked poses in Blue
                    axes.paint_uniform_color([0, 0, 1])
                    poses_o3d.append(axes)

            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            o3d.visualization.draw_geometries([origin] + poses_o3d)
            
            
            
            # exit(0)
        
            print("Reached end frame ID")
            
            
            # Visualize accumulated point clouds
            poses_o3d = []
            

            pcd_list = [accumulated_pc.get_full_pointcloud(fraction = 0.1, voxel_size=0.2)]

            o3d.visualization.draw_geometries(pcd_list + poses_o3d + [origin])

            print("SLAM: ", slam)   
            print("Tracker: ", slam.tracker)
            print("Map: ", slam.map)
            # PRint number of keyframes and number of points in the map in global and local maps
            print("Number of keyframes in global map: ", len(slam.map.keyframes))
            print("Number of points in global map: ", len(slam.map.map_points))



            break