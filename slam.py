#!/usr/bin/env python3
from config import Config
from io_utils.dataset import dataset_factory, SensorType
from io_utils.ground_truth import groundtruth_factory
import logging 
from utilities.utils_depth import depth2pointcloud
from utilities.utils_metrics import estimate_error_R_T
import cv2
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint
import torch
import numpy as np
from utilities.utils_draw import draw_torch_image, visualize_global_map
from collections import deque
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader
from core.slam_system import SLAMSystem as SLAM
import open3d as o3d
from utilities.utils_misc import remove_duplicates_from_index_arrays, delaunay_with_kps_new
from utilities.utils_draw import visualize_matches
from utilities.utils_edges import (find_matching_edges, find_dynamic_edges, 
                                 find_connected_components, create_dynamic_mask,
                                 visualize_edges, visualize_dynamic_components)

def create_edge_pairs(simplicies):
    """Create edge pairs from Delaunay triangulation simplicies."""
    edge_pairs = []
    for simplex in simplicies:
        a, b, c = simplex
        for edge in [(a,b), (b,c), (c,a)]:
            if edge not in edge_pairs and (edge[1], edge[0]) not in edge_pairs:
                edge_pairs.append(edge)
    return edge_pairs

def analyze_edges(frame1, frame2, matches, prev_edges=None):
    """Analyze edges between two frames, returns dynamic edges."""
    idxs_ref, idxs_cur = matches[:, 0], matches[:, 1]
    idxs_ref, idxs_cur = remove_duplicates_from_index_arrays(idxs_ref, idxs_cur)
    
    # Create mappings between indices
    idxs_ref_dict = dict(zip(idxs_ref, idxs_cur))
    idxs_cur_dict = dict(zip(idxs_cur, idxs_ref))
    
    # If no previous edges, create from Delaunay
    if prev_edges is None:
        prev_tri_simplicies, _, _ = delaunay_with_kps_new(
            frame1.img, frame1.keypoints, idxs_ref)
        prev_edges = create_edge_pairs(prev_tri_simplicies)
    
    # Map edges to current frame
    curr_edges = []
    prev_mapped_edges = []
    
    for edge in prev_edges:
        a, b = edge
        if a in idxs_ref_dict and b in idxs_ref_dict:
            curr_a = idxs_ref_dict[a]
            curr_b = idxs_ref_dict[b]
            curr_edges.append((curr_a, curr_b))
            prev_mapped_edges.append((a, b))
    
    # Calculate edge lengths and identify dynamic edges
    dynamic_edges = []
    if curr_edges:
        lengths_prev = [np.linalg.norm(frame1.keypoints[a] - frame1.keypoints[b]) 
                       for a, b in prev_mapped_edges]
        lengths_curr = [np.linalg.norm(frame2.keypoints[a] - frame2.keypoints[b]) 
                       for a, b in curr_edges]
        
        # Identify dynamic edges (threshold from slam_backup.py)
        dynamic_threshold = 4  # pixels
        dynamic_edges = [i for i, (l1, l2) in enumerate(zip(lengths_prev, lengths_curr)) 
                        if abs(l1 - l2) > dynamic_threshold]
    
    return curr_edges, dynamic_edges

def find_connected_components(edges, dynamic_edges):
    """Find connected components excluding dynamic edges."""
    # Create adjacency matrix
    graph = {}
    for i, (a, b) in enumerate(edges):
        if i not in dynamic_edges:
            if a not in graph: graph[a] = []
            if b not in graph: graph[b] = []
            graph[a].append(b)
            graph[b].append(a)
    
    # DFS to find components
    def dfs(node, visited, component):
        visited[node] = True
        component.append(node)
        for neighbor in graph.get(node, []):
            if not visited.get(neighbor, False):
                dfs(neighbor, visited, component)
                
    visited = {}
    components = []
    for node in graph:
        if not visited.get(node, False):
            component = []
            dfs(node, visited, component)
            if len(component) > 2:  # Only keep components with more than 2 points
                components.append(component)
    
    return sorted(components, key=len, reverse=True)

def visualize_dynamic_analysis(frame1, frame2, edges, dynamic_edges, components=None):
    """Visualize dynamic edge analysis results."""
    stacked = np.hstack([frame1.img, frame2.img])
    w = frame1.img.shape[1]
    
    # Draw edges (green=static, red=dynamic)
    for i, (a, b) in enumerate(edges):
        color = (0, 0, 255) if i in dynamic_edges else (0, 255, 0)
        # Draw in first frame
        pt1 = tuple(map(int, frame1.keypoints[a]))
        pt2 = tuple(map(int, frame1.keypoints[b]))
        cv2.line(stacked, pt1, pt2, color, 2)
        # Draw in second frame (offset by width)
        pt1 = (int(frame2.keypoints[a][0] + w), int(frame2.keypoints[a][1]))
        pt2 = (int(frame2.keypoints[b][0] + w), int(frame2.keypoints[b][1]))
        cv2.line(stacked, pt1, pt2, color, 2)
    
    # Draw components if available
    if components:
        colors = [(255,0,0), (0,255,255), (255,0,255)]  # Different colors for top components
        for idx, comp in enumerate(components[:3]):  # Show top 3 components
            color = colors[idx % len(colors)]
            # Draw convex hulls
            pts = np.array([frame2.keypoints[i] for i in comp], dtype=np.int32)
            hull = cv2.convexHull(pts)
            # Draw in second frame (offset by width)
            hull_offset = hull.copy()
            hull_offset[:,:,0] += w
            cv2.polylines(stacked, [hull_offset], True, color, 2)
    
    cv2.imshow("Dynamic Analysis", stacked)
    cv2.waitKey(1)
    
    return stacked


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
    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Load groundtruth trajectory if available
    if groundtruth is not None:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        if gt_poses is None or len(gt_poses) == 0:
            print("Error: No ground truth poses loaded!")
            exit(1)

    # Initialize processing variables
    starting_img_id = config.start_frame_id
    ending_img_id = config.end_frame_id
    img_id = starting_img_id
    global_poses = {}
    accumulated_clouds = {}
    invalid_frames = set()
    skipped_frames = set()
    # Short_trajectory Will contain the last few selected frames as a queue - this will have frames from ::: curr_frame - kNumFramesAway to curr_frame - 1
    short_trajectory = deque(maxlen=kNumFramesAway) 
    prev_edges = None  # Store edges from previous analysis
    last_analysis_id = starting_img_id  # Track when we last did analysis
    last_analysis_frame = None
    
    while True: 
        # Get current frame data
        if dataset.isOk(): 
            img = dataset.getImage(img_id)
            depth_img = dataset.getDepth(img_id) * depth_scale
            
            if img is not None:
                timestamp = dataset.getTimestamp()
                point_cloud = depth2pointcloud(
                    depth_img, img,
                    config.cam_settings["Camera.fx"], 
                    config.cam_settings["Camera.fy"],
                    config.cam_settings["Camera.cx"], 
                    config.cam_settings["Camera.cy"],
                    max_depth=100000.0, min_depth=0.0
                )
                
                # Convert to torch tensor
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                
                # # Store point cloud if valid
                # accumulated_clouds[img_id] = point_cloud
                
                # Get pose from groundtruth if available
                if groundtruth is not None and gt_poses is not None and gt_timestamps is not None:
                    closest_idx = np.argmin(np.abs(gt_timestamps - timestamp))
                    time_diff = abs(gt_timestamps[closest_idx] - timestamp)
                    
                    if time_diff > 0.1:
                        print(f"Warning: Large time diff ({time_diff}s) - skipping frame {img_id}")
                        skipped_frames.add(img_id)
                    else:
                        global_poses[img_id] = gt_poses[closest_idx]

                else:
                    print("No ground truth poses available")
                # Visualize current frame
                draw_torch_image(img_tensor)
                cv2.waitKey(1)

                # TODO: Implement SLAM pipeline here
                # If first frame, initialize slam
                if img_id == starting_img_id:
                    # 1. Create a slam object which contains map object, slam_parameters_loaded, Gaussian Model, tracker object, extractor object, matcher object
                    # 2. Initialize the frame object with the current frame, camera matrix, pose, depth, image, keypoints, descriptors
                    # 3. Make it a keyframe and add it to the map object
                    # 4. Map object will have local and global map points, local map points will only contain points from the last kNumKFsLM frames
                    # 5. Global map points will contain all the map points, initialize the tracker object for slam object as well. 
                    # 6. SLAM object is common through out the slam session 
                    # 7. Tracker object is common through out the slam session
                    # 8. Local and Global map for the first frame is initialized with the detected kps and for the gaussian model, optimize it with the full image with the dynamic mask. incase the mask exists. if not normal gaussian splatting is applied.
                    # 9. The first frame is added to the short_trajectory as well.
                    # 10. The first frame is added to the map object as well.
                    # 11. The first frame is added to the global map as well.
                    # 12. The first frame is added to the local map as well.
                    
                    print("Initializing SLAM")
                    print("Processing frame: ", img_id)
                    curr_frame = get_frame_from_pyslam_dataloader(dataset, groundtruth, img_id, config)
                    slam.initialize(curr_frame)
                    short_trajectory.append(curr_frame)
                    print("Initialized SLAM")
                    print("SLAM: ", slam)
                    print("Tracker: ", slam.tracker)
                    print("Map: ", slam.map)
                    print("Short Trajectory: ", short_trajectory)
                    print("Processing frame: ", img_id)


                    

                    if img_id in global_poses: # SETTING POSE TO GT
                        print("Setting pose of frame : ", img_id, " to GT")
                        # curr_frame.pose = global_poses[img_id]

                    


                    prev_frame = curr_frame
                    last_analysis_frame = curr_frame
                    
                else: 
                    # 1. Create a frame object with the current frame, camera matrix, pose, depth, image, keypoints, descriptors
                    # 2. Frame object will have the current frame, camera matrix, pose, depth, image, keypoints, descriptors
                    # 3. Check if the frame is a keyframe or not, if it is a keyframe, add it to the map object, if not, track the frame with the previous keyframe which can be found in the short_trajectory
                    # 4. If the frame is not a keyframe, track the frame with the previous keyframe, if the tracking is successful, save the pose
                    # 5. If the tracking is not successful, mark the frame as invalid and skip it 
                    # 6. We track the common # matches with the last keyframe, if it goes below a certain threshold, we mark the frame as a new keyframe. 
                    # 7. If the frame is a keyframe, we optimize the map points with the new frame, and update the local and global map points. 
                    # 8. We also update the short_trajectory with the new keyframe. 
                    print("Processing frame: ", img_id)
                    curr_frame = get_frame_from_pyslam_dataloader(dataset, groundtruth, img_id, config)
                    short_trajectory.append(curr_frame)
                    
                    ## FEW STEPS OF DELAUNAY GIVES US PROMPTS FOR SAM2 
                    dynamic_mask = None
                    curr_frame._dynamic_mask = dynamic_mask

                    # Track frame relative to previous keyframe
                    print("Tracking frame: ", img_id, " with previous keyframe")
                    tracking_status = slam.track_frame(curr_frame, dynamic_mask=dynamic_mask)
                    print("Tracking status: ", tracking_status)

                    ## Update the pose of the current frame with the ground truth pose if available
                    if img_id in global_poses: ## SETTING POSE TO GT
                        print("Setting pose of frame : ", img_id, " to GT")
                        # curr_frame.pose = global_poses[img_id]
                    
                    ## ERROR CALUCLATION
                    print("Prev frame pose: ", prev_frame.pose)
                    print("Curr frame pose: ", curr_frame.pose)
                    relative_pose = np.linalg.inv(prev_frame.pose) @ curr_frame.pose 
                    relative_pose_gt = np.linalg.inv(global_poses[prev_frame.id]) @ global_poses[curr_frame.id]
                    # ERROR IN RELATIVE POSE CALCULATION
                    angle_err, t_err = estimate_error_R_T(relative_pose, relative_pose_gt)
                    print("Local Rotation error (degrees):", angle_err)
                    print("Local Translation error (meters):", t_err)

                    absolute_pose = global_poses[starting_img_id] @ curr_frame.pose
                    absolute_pose_gt = global_poses[curr_frame.id]
                    # ERROR IN ABSOLUTE POSE CALCULATION
                    angle_err, t_err = estimate_error_R_T(absolute_pose, absolute_pose_gt)
                    print("Global Rotation error (degrees):", angle_err)
                    print("Global Translation error (meters):", t_err)


                    prev_frame = curr_frame

                    # Analyze dynamic objects every kNumFramesAway frames
                    if img_id - last_analysis_id >= kNumFramesAway:
                        ref_frame = short_trajectory[0]  # Get frame from kNumFramesAway ago
                        
                        # Match frames
                        matches = slam.tracker.match_frames(ref_frame, curr_frame)
                        if matches is not None and len(matches) > 10:  # Ensure enough matches
                            # Analyze edges and find dynamic objects
                            prev_edges, curr_edges = find_matching_edges(ref_frame, curr_frame, matches)
                            
                            if prev_edges:
                                # Find dynamic edges
                                dynamic_edges = find_dynamic_edges(ref_frame, curr_frame, prev_edges, curr_edges)
                                
                                # Find connected components
                                components = find_connected_components(curr_edges, dynamic_edges)
                                
                                # Create dynamic mask
                                dynamic_mask = create_dynamic_mask(curr_frame, components)
                                curr_frame._dynamic_mask = dynamic_mask
                                
                                # Visualize results
                                if config.ShowDebugImages:
                                    visualize_edges(ref_frame, curr_frame, prev_edges, curr_edges, 
                                                 dynamic_edges, components)
                                    # Add new visualization
                                    visualize_dynamic_components(ref_frame, curr_frame, components)
                                    
                                last_analysis_id = img_id
                                last_analysis_frame = curr_frame
                    
                    # Track frame with dynamic mask
                    tracking_status = slam.track_frame(curr_frame, 
                                                    dynamic_mask=getattr(curr_frame, '_dynamic_mask', None))
                    
                    # ...rest of existing tracking code...

        # Process next frame
        img_id += 1
        
        if img_id > ending_img_id:
            print("Reached end frame ID")
            print("Visualizing final map...")
            
            # Save the map
            saved_map_path = slam.map.save()
            print(f"Map saved to: {saved_map_path}")
            
            # Visualize only the map points and keyframe trajectory
            visualize_global_map(slam.map,  
                               title=f"Final Map - {len(slam.map.keyframes)} keyframes", 
                               dense=True)
            

            
            # Visualize accumulated point clouds
            '''
            pcd_list = []
            for i in range(len(global_poses)):
                if i in accumulated_clouds:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(accumulated_clouds[i].points)
                    pcd.colors = o3d.utility.Vector3dVector(accumulated_clouds[i].colors)
                    # transform point cloud to global pose
                    pose = global_poses[i]
                    pcd.transform(pose)
                    pcd_list.append(pcd)
                    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
                    axis.transform(pose)
                    pcd_list.append(axis)
            o3d.visualization.draw_geometries(pcd_list)
            '''
                    
            break