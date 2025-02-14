#!/usr/bin/env python3
from config import Config
from io_utils.dataset import dataset_factory, SensorType
from io_utils.ground_truth import groundtruth_factory
import logging 
from utilities.utils_depth import depth2pointcloud
import cv2
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from thirdparty.LightGlue.lightglue import viz2d
from thirdparty.LightGlue.lightglue.utils import rbd
import torch
from config_parameters import Parameters  
import numpy as np
from utilities.utils_draw import draw_torch_image, visualize_matches
from utilities.utils_misc import estimate_pose_ransac, estimate_pose_icp
import open3d as o3d
from utilities.utils_misc import *

if __name__ == "__main__":
    config = Config()
    dataset = dataset_factory(config) 
    depth_scale = 1/5000.0 # mm to meters

    groundtruth = groundtruth_factory(config.dataset_settings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = 5000
    if config.num_features_to_extract > 0:
        num_features = config.num_features_to_extract  

    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
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

    # Processing the dataset 
    starting_img_id = 215# 215 is close to human entrance
    img_id = starting_img_id

    global_poses = {}
    accumulated_clouds = {}
    invalid_frames = set()
    skipped_frames = set()  # Track frames we skip due to missing poses

    while True: 


        img, depth_img = None, None 

        # Check if dataset is ok
        if dataset.isOk(): 
            logging.debug("dataset is ok") 
            img = dataset.getImage(img_id)
            img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
            depth_img = dataset.getDepth(img_id) * depth_scale # Some scaling
            print("depth_img: ", depth_img[50])
            print("Max depth: ", np.max(depth_img))
            logging.debug("img_id: %d", img_id)

        if img is not None:
            timestamp = dataset.getTimestamp() 
            next_timestamp = dataset.getNextTimestamp() 
            frame_duration = next_timestamp - timestamp if (timestamp is not None and next_timestamp is not None) else -1.0 
            logging.debug("image with id %d has timestamp %f and next_timestamp %f, frame_duration: %f", img_id, timestamp, next_timestamp, frame_duration)
            logging.debug("logging data associated to id %d to rerun", img_id)

            point_cloud = depth2pointcloud(depth_img, img, 
                                       config.cam_settings["Camera.fx"], config.cam_settings["Camera.fy"], 
                                       config.cam_settings["Camera.cx"], config.cam_settings["Camera.cy"], 
                                       max_depth=100000.0, min_depth=0.0)
            
            print("Shape of img1", img.shape)
            # Convert to torch tensor  3 * H * W
            img = torch.from_numpy(img).permute(2, 0, 1).float()/ 255.0
            print("Shape of img2: ", img.shape)
            
            
        # Show image    
        if img is not None:
            draw_torch_image(img)
        else:
            logging.debug("img is None")

        if img is not None:
            accumulated_clouds[img_id] = point_cloud

        # Comparing [0-N, 1-N+1, 2-2+N, 3-3+N, .....]
        if img_id > starting_img_id + Parameters.kNumFramesAway - 1:
            
            prev_img = dataset.getImage(img_id - 1)
            prev_depth_img = dataset.getDepth(img_id - 1) * depth_scale
            prev_point_cloud = depth2pointcloud(prev_depth_img, prev_img,
                                        config.cam_settings["Camera.fx"], config.cam_settings["Camera.fy"], 
                                        config.cam_settings["Camera.cx"], config.cam_settings["Camera.cy"],
                                        max_depth=100000.0, min_depth=0.0)
            
            prev_img = torch.from_numpy(prev_img).permute(2, 0, 1).float() / 255.0
            prev_depth_img = torch.from_numpy(prev_depth_img.astype(np.float32)).unsqueeze(0)


            if Parameters.kNumFramesAway != 1: 
                ref_prev_same = False
                ref_img = dataset.getImage(img_id - Parameters.kNumFramesAway)
                ref_depth_img = dataset.getDepth(img_id - Parameters.kNumFramesAway) * depth_scale
                ref_point_cloud = depth2pointcloud(ref_depth_img, ref_img, 
                                       config.cam_settings["Camera.fx"], config.cam_settings["Camera.fy"], 
                                       config.cam_settings["Camera.cx"], config.cam_settings["Camera.cy"], 
                                       max_depth=10.0, min_depth=0.0)
            
                ref_img = torch.from_numpy(ref_img).permute(2, 0, 1).float() / 255.0
                ref_depth_img = torch.from_numpy(ref_depth_img.astype(np.float32)).unsqueeze(0)

            else:
                ref_prev_same = True
                ref_img = prev_img
                ref_depth_img = prev_depth_img
                ref_point_cloud = prev_point_cloud

            cv2.imshow("Ref_depth_image", ref_depth_img.permute(1, 2, 0).cpu().numpy()) 
            cv2.waitKey(2)
            print("ref_depth_img: ", ref_depth_img)
                
            
            # Matching
            ref_feat = extractor.extract(ref_img.to(device))
            curr_feat = extractor.extract(img.to(device))
            matches01 = matcher({"image0": ref_feat, "image1": curr_feat})
            feats0, feats1, matches01 = [
                rbd(x) for x in [ref_feat, curr_feat, matches01]
            ]  # remove batch dimension

            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
            
            # Print shape of every variable
            ''' 
            Shape of img1 (480, 640, 3)
            Shape of img2:  torch.Size([3, 480, 640])
            ref_img:  torch.Size([3, 480, 640])
            ref_depth_img:  torch.Size([1, 480, 640])
            ref_feat:  torch.Size([1, 1370, 2])
            curr_feat:  torch.Size([1, 1338, 2])
            matches01:  torch.Size([589, 2])
            feats0:  torch.Size([1370, 2])
            feats1:  torch.Size([1338, 2])
            matches:  torch.Size([589, 2])
            kpts0:  torch.Size([1370, 2])
            kpts1:  torch.Size([1338, 2])
            m_kpts0:  torch.Size([589, 2])
            m_kpts1:  torch.Size([589, 2])
            ref_point_cloud:  (257067, 3)
            '''

            # Plot matches on the image 
            matched_image = visualize_matches(ref_img, img, kpts0, kpts1, matches)

            cv2.imshow("Matched Image", matched_image)
            cv2.waitKey(2)


            # Convert to numpy for OpenCV
            ref_depth_np = ref_depth_img.squeeze().cpu().numpy() if isinstance(ref_depth_img,
                                                                               torch.Tensor) else ref_depth_img
            ref_img_np = ref_img.permute(1, 2, 0).cpu().numpy() if isinstance(ref_img, torch.Tensor) else ref_img

            # Camera matrix (intrinsics)
            camera_matrix = np.array([
                [config.cam_settings["Camera.fx"], 0, config.cam_settings["Camera.cx"]],
                [0, config.cam_settings["Camera.fy"], config.cam_settings["Camera.cy"]],
                [0, 0, 1]
            ], dtype=np.float64)

            # Estimate Pose (RANSAC + PnP)
            # Convert to numpy for OpenCV
            kpts0_np = kpts0.cpu().numpy()
            kpts1_np = kpts1.cpu().numpy()
            m_kpts0_np = m_kpts0.cpu().numpy()
            m_kpts1_np = m_kpts1.cpu().numpy()
            matches_np = matches.cpu().numpy()

            '''
            success, R, t, inliers = estimate_pose_ransac(ref_depth_np,ref_img_np, kpts0_np, kpts1_np, matches_np, camera_matrix)

            if not success:
                print("Error in pose estimation: ", img_id)
                print("R: ", R)
                print("t: ", t)
                success = False
                # Stop the loop
                break
            '''


            # TRACK POSE
            '''
            T = estimate_pose_icp(prev_point_cloud.points, point_cloud.points)
            # Add to global poses
            global_poses[img_id] = T @ global_poses[img_id - 1]
            print("Global pose at ", img_id, ":\n", global_poses[img_id])
            '''

            # For now using GT poses for global poses. 
            # Get pose from groundtruth (timestamp, x,y,z, qx,qy,qz,qw, scale)
            timestamp = dataset.getTimestamp()
            if groundtruth is not None and gt_poses is not None and gt_timestamps is not None:
                print(f"Looking for pose at timestamp: {timestamp}")
                closest_idx = np.argmin(np.abs(gt_timestamps - timestamp))
                time_diff = abs(gt_timestamps[closest_idx] - timestamp)
                
                if time_diff > 0.1:  # More than 100ms difference
                    print(f"Warning: Large time difference ({time_diff}s) - skipping frame {img_id}")
                    skipped_frames.add(img_id)
                    img_id += 1
                    continue  # Skip this frame entirely
                else:
                    T = gt_poses[closest_idx]
                    print(f"Found pose at timestamp: {gt_timestamps[closest_idx]}")
                    print(f"Pose matrix:\n{T}")
                    # Only add pose if we have a valid one
                    global_poses[img_id] = T
                    # Only store point cloud if we have a valid pose
                    if img is not None:
                        accumulated_clouds[img_id] = point_cloud
            else:
                print("Error: No ground truth available - cannot continue")
                break  # Stop processing if no ground truth is available

            # TRACK DYNAMIC OBJECTS
            # print("Matches: ", matches_np)
            # print("m_kpts0_np: ", m_kpts0_np)


            idxs_ref, idxs_curr = matches_np[:, 0], matches_np[:, 1]
            print("idxs_ref: ", len(idxs_ref))
            idxs_ref, idxs_curr = remove_duplicates_from_index_arrays(idxs_ref, idxs_curr)
            print("idxs_ref after removing duplicates: ", len(idxs_ref))  

            if len(idxs_curr) > 3 and len(idxs_ref) > 3:
                # Now applying Delaunay triangulation on the matched keypoints 
                _, _, prev_delaunay_img = delaunay_with_kps_new(ref_img_np, kpts0_np, idxs_ref)
                _, _, curr_delaunay_img = delaunay_with_kps_new(img.permute(1, 2, 0).cpu().numpy(), kpts1_np, idxs_curr)
                if Parameters.kShowDebugImages:
                    delaunay_visualization(prev_delaunay_img, curr_delaunay_img)
                    # delaunay_visualization(c_prev_delaunay_img, c_curr_delaunay_img)      

            
            
            '''
            if success:
                print("Pose estimation successful!")
                print("Rotation:\n", R)
                print("Translation:\n", t)
                # print("Inliers:", inliers)
                # --- Accumulate pose (as shown in previous example) ---

                # Example of how to visualize only inlier matches:
                if inliers is not None:
                  inlier_matches = matches[inliers.ravel()]
                  matched_image_inliers = visualize_matches(ref_img, img, kpts0, kpts1, inlier_matches)
                  cv2.imshow("Inlier Matches", matched_image_inliers)

            else:
                print("Pose estimation failed!")
            '''


        
        if img_id > starting_img_id + 15: 
            # Create visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window()

            # Create a single point cloud from all frames
            global_map = o3d.geometry.PointCloud()
            all_points = []
            all_colors = []

            # Transform and combine all point clouds
            for frame_id, points in accumulated_clouds.items():
                if frame_id in skipped_frames or frame_id in invalid_frames:
                    continue
                if frame_id not in global_poses:
                    continue
                if np.array_equal(global_poses[frame_id], np.eye(4)):  # Skip identity poses
                    continue
                    
                # Get points and colors
                pts = points.points
                colors = points.colors
                
                # Convert points to homogeneous coordinates
                pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))
                
                # Transform points using global pose
                transformed_pts = (global_poses[frame_id] @ pts_homog.T).T[:, :3]
                
                all_points.append(transformed_pts)
                all_colors.append(colors)

            # Combine all points into single point cloud
            if all_points:
                global_map.points = o3d.utility.Vector3dVector(np.vstack(all_points))
                global_map.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
                
                # Optional: Downsample to reduce density
                global_map = global_map.voxel_down_sample(voxel_size=0.05)
                
                # Add point cloud to visualizer
                vis.add_geometry(global_map)

                # Add camera poses as coordinate frames
                for frame_id, pose in global_poses.items():
                    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    cam_frame.transform(pose)
                    vis.add_geometry(cam_frame)

                # Add trajectory visualization
                points = []
                for pose in global_poses.values():
                    points.append(pose[:3, 3])

                if points:
                    # Create trajectory line set
                    trajectory = o3d.geometry.LineSet()
                    trajectory.points = o3d.utility.Vector3dVector(points)
                    trajectory.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(points)-1)])
                    trajectory.paint_uniform_color([1, 0, 0])  # Red color
                    vis.add_geometry(trajectory)

                # Add world coordinate frame
                world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
                vis.add_geometry(world_frame)

                # Set view control
                vis.get_view_control().set_zoom(0.5)
                vis.get_view_control().set_front([0, 0, -1])
                vis.get_view_control().set_up([0, -1, 0])

                # Run visualizer
                vis.run()
                vis.destroy_window()
            break

        img_id += 1