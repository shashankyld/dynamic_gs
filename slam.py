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
import matplotlib.pyplot as plt
import networkx as nx

def create_histogram_image(data, num_bins=50, value_range=(-20, 20), width=640, height=480):
    """Create a histogram visualization using OpenCV"""
    # Calculate histogram
    hist_counts, bin_edges = np.histogram(data, bins=num_bins, range=value_range)
    
    # Normalize histogram to fit in image
    hist_counts = hist_counts / hist_counts.max() * (height - 60)  # Leave margin for labels
    
    # Create image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw histogram bars
    bin_width = int(width / num_bins)
    for i in range(num_bins):
        x1 = i * bin_width
        y1 = height - int(hist_counts[i]) - 30  # Leave space for x-axis labels
        x2 = (i + 1) * bin_width - 1
        y2 = height - 30
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        # Draw vertical grid lines
        cv2.line(img, (x1, 0), (x1, height-30), (200, 200, 200), 1)
    
    # Draw horizontal grid lines (5 lines)
    for i in range(5):
        y = int(i * (height-30) / 4)
        cv2.line(img, (0, y), (width, y), (200, 200, 200), 1)
    
    # Draw axes
    cv2.line(img, (0, height-30), (width, height-30), (0,0,0), 2)  # X-axis
    cv2.line(img, (0, 0), (0, height-30), (0,0,0), 2)  # Y-axis
    
    # Add x-axis labels (every 5th bin)
    for i in range(0, num_bins, 5):
        x_val = value_range[0] + (value_range[1] - value_range[0]) * i / num_bins
        x_pos = i * bin_width
        cv2.putText(img, f"{x_val:.1f}", (x_pos-20, height-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    
    # Add title
    cv2.putText(img, "Edge Length Differences", (width//3, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    return img

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
            # print("depth_img: ", depth_img[50])
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

            print("Number of keypoints in ref image: ", len(kpts0))
            print("Number of keypoints in curr image: ", len(kpts1))
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
            print("kpts0_np: ", kpts0_np)   
            print("kpts1_np: ", kpts1_np)
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
            # Create a two dictionaries with idxs of ref and curr images mapped to each other
            idxs_ref_dict = dict(zip(idxs_ref, idxs_curr))
            idxs_curr_dict = dict(zip(idxs_curr, idxs_ref))
            print("idxs_ref_dict: ", len(idxs_ref_dict))
            print("idxs_curr_dict: ", len(idxs_curr_dict))
            print("idxs_ref: ", len(idxs_ref))
            print("idxs_curr: ", len(idxs_curr))
            idxs_ref, idxs_curr = remove_duplicates_from_index_arrays(idxs_ref, idxs_curr)
            print("idxs_ref after removing duplicates: ", len(idxs_ref))  
            print("Len of kpts0_np: ", len(kpts0_np))
            print("Len of idxs_ref: ", len(idxs_ref))
            if len(idxs_curr) > 3 and len(idxs_ref) > 3:
                # Now applying Delaunay triangulation on the matched keypoints 
                prev_tri_simplicies, prev_tri_vertices, prev_delaunay_img = delaunay_with_kps_new(ref_img_np, kpts0_np, idxs_ref)
                curr_tri_simplicies, curr_tri_vertices, curr_delaunay_img = delaunay_with_kps_new(img.permute(1, 2, 0).cpu().numpy(), kpts1_np, idxs_curr)
                if Parameters.kShowDebugImages:
                    delaunay_visualization(prev_delaunay_img, curr_delaunay_img)
            

            def create_idxs_pairs(simplicies):
                idxs_pairs = []
                for simplex in simplicies:
                    a,b,c = simplex
                    for i in range(3):
                        # If (a,b) or (b,a) is not in the list, add it; else skip
                        if (a,b) not in idxs_pairs and (b,a) not in idxs_pairs:
                            idxs_pairs.append((a,b))
                        if (b,c) not in idxs_pairs and (c,b) not in idxs_pairs:
                            idxs_pairs.append((b,c))
                        if (c,a) not in idxs_pairs and (a,c) not in idxs_pairs:
                            idxs_pairs.append((c,a))
                return idxs_pairs

            # Create a dict of idxs as keys with connected idxs as values for both the delaunay triangles
            prev_idxs_pairs = create_idxs_pairs(prev_tri_simplicies)
            curr_idxs_pairs = create_idxs_pairs(curr_tri_simplicies)
            # print("prev_idxs_dict: ", prev_idxs_pairs)
            # print("curr_idxs_dict: ", curr_idxs_pairs)
            print("prev_idxs_pairs: ", len(prev_idxs_pairs))
            print("curr_idxs_pairs: ", len(curr_idxs_pairs))

            ## Finding common pairs in both the images
            # Convert prev_idxs_pairs to be able to map with curr_idxs_pairs
            # print("idxs_ref_dict: ", idxs_ref_dict.keys())
            # print("idxs_curr_dict: ", idxs_curr_dict.keys())
            # print("prev_idxs_pairs: ", prev_idxs_pairs)
            prev_converted_idxs_pairs = []
            for idx_pair in prev_idxs_pairs:
                a,b = idx_pair
                if a in idxs_ref_dict and b in idxs_ref_dict:
                    prev_converted_idxs_pairs.append((idxs_ref_dict[a], idxs_ref_dict[b]))
                else:
                    "This is not possible because we only have matched keypoints" 
                    # Error and break the program - catch error
                    break

            # print("prev_converted_idxs_pairs: ", prev_converted_idxs_pairs)
            print("prev_converted_idxs_pairs: ", len(prev_converted_idxs_pairs))
            common_pairs = []
            for idx_pair in prev_converted_idxs_pairs:
                # Check for both (a,b) and (b,a) in curr_idxs_pairs - if one of them is present, add to common_pairs
                a,b = idx_pair
                if (a,b) in curr_idxs_pairs or (b,a) in curr_idxs_pairs:
                    common_pairs.append((a,b))
            # print("common_pairs: ", common_pairs)
            print("common_pairs: ", len(common_pairs))

            # Common pairs in previous using the dictionary
            common_pairs_prev = []
            for pair in common_pairs:
                a,b = pair
                # Convert to prev_idxs using the dictionary idxs_curr_dict
                a_prev = idxs_curr_dict[a]
                b_prev = idxs_curr_dict[b]
                common_pairs_prev.append((a_prev, b_prev))
            print("common_pairs_prev: ", len(common_pairs_prev))





            print("prev_tri_indices: ", prev_tri_simplicies[0])
            print("prev_tri_vertices: ", prev_tri_vertices[0])
            print("curr_tri_indices: ", curr_tri_simplicies[0])
            print("curr_tri_vertices: ", curr_tri_vertices[0])
            # Print max value in the indices, vertices
            print("Max value in prev_tri_indices: ", np.max(prev_tri_simplicies))
            print("Max value in prev_tri_vertices: ", np.max(prev_tri_vertices))
            print("Max value in curr_tri_indices: ", np.max(curr_tri_simplicies))
            print("Max value in curr_tri_vertices: ", np.max(curr_tri_vertices))


            def visualize_common_edges(img, kpts, common_pairs):
                img = img.copy()
                for pair in common_pairs:
                    a,b = pair
                    cv2.line(img, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])), (0,255,0), 2)
                return img

            # Visualize common edges
            prev_img_with_edges = visualize_common_edges(ref_img_np, kpts0_np, common_pairs_prev)
            curr_img_with_edges = visualize_common_edges(img.permute(1, 2, 0).cpu().numpy(), kpts1_np, common_pairs)
            cv2.imshow("Prev_img_with_edges", prev_img_with_edges)
            cv2.imshow("Curr_img_with_edges", curr_img_with_edges)
            cv2.waitKey(2)


            # Estimate Lengths of these common edges in both the images 

            lengths_prev = []
            lengths_curr = []
            for i in range(len(common_pairs_prev)):
                # We have equal number of common pairs in both the images
                a,b = common_pairs_prev[i]
                lengths_prev.append(np.linalg.norm(kpts0_np[a] - kpts0_np[b]))
                a,b = common_pairs[i]
                lengths_curr.append(np.linalg.norm(kpts1_np[a] - kpts1_np[b]))

            print("Differences in lengths: ", np.array(lengths_prev) - np.array(lengths_curr))
            # Plot these differences in lengths as a histogram - update the plot with the loop
            import matplotlib.pyplot as plt
            # Update the plot
            
            if len(common_pairs_prev) > 0:
                # Calculate length differences
                length_diffs = np.array(lengths_prev) - np.array(lengths_curr)

                
                # Create and show histogram
                hist_img = create_histogram_image(length_diffs, num_bins=50)
                cv2.imshow("Edge Length Differences Histogram", hist_img)
                cv2.waitKey(2)

            
            # Remove the edges which are changing lengths significantly with a threshold
            dynamic_threshold = 4
            dynamic_edges = []
            for i in range(len(lengths_prev)):
                if abs(lengths_prev[i] - lengths_curr[i]) > dynamic_threshold:
                    dynamic_edges.append(i)
            print("Dynamic edges: ", dynamic_edges)

            # Visualize dynamic edges

            def visualize_dynamic_edges(img, kpts, common_pairs, dynamic_edges, only_static=False):
                img = img.copy()
                # Add number of static and dynamic edges on top rigt
                cv2.putText(img, f"Static: {len(common_pairs) - len(dynamic_edges)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(img, f"Dynamic: {len(dynamic_edges)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if only_static:
                    # Only visualize static edges in green
                    for i in range(len(common_pairs)):
                        if i not in dynamic_edges:
                            a,b = common_pairs[i]
                            cv2.line(img, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])), (0,255,0), 2)
                            
                    return img
                # common edges in green and if they are dynamic, in red
                for i in range(len(common_pairs)):
                    a,b = common_pairs[i]
                    if i in dynamic_edges:
                        cv2.line(img, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])), (0,0,255), 2)
                    else:
                        cv2.line(img, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])), (0,255,0), 2)
                return img
            


        

            img = dataset.getImage(img_id)
            img = torch.from_numpy(img).permute(2, 0, 1).float()/ 255.0
            curr_img_np = img.permute(1, 2, 0).cpu().numpy()
            curr_img_with_dynamic_edges = visualize_dynamic_edges(curr_img_np, kpts1_np, common_pairs, dynamic_edges, only_static=False)

            curr_img_with_dynamic_edges2 = visualize_dynamic_edges(curr_img_np, kpts1_np, common_pairs, dynamic_edges, only_static=True)
            cv2.imshow("Curr_img_with_dynamic_edges", curr_img_with_dynamic_edges)
            cv2.imshow("Curr_img_with_dynamic_edges2", curr_img_with_dynamic_edges2)
            cv2.waitKey(2)

            common_pais_after_removing_dynamic_edges = [pair for i, pair in enumerate(common_pairs) if i not in dynamic_edges]
            common_pairs = common_pais_after_removing_dynamic_edges ### MAJOR ADAPTATION IN NAMING, CAN BE CONFUSING

            ## Now, we will divide the whole graph into connected components using graph traversal and find the largest connected component,
            ## Added thing we can do is - if more connected components emerge, we can see consensus between these sub graphs
            ## Estimate distance between randomly picked point in two different connected components and see if they are changing - if they are not changing, we can say that they are together.
            ## At the end, the biggest connected component along with its consensus subgraphs are considered static 
            ## The rest are dynamic
                

            ##TODO: Implement the above logic
            # Create a graph with the common pairs
            graph_adj_mat = {}
            for pair in common_pairs:
                a,b = pair
                if a not in graph_adj_mat:
                    graph_adj_mat[a] = []
                if b not in graph_adj_mat:
                    graph_adj_mat[b] = []
                graph_adj_mat[a].append(b)
                graph_adj_mat[b].append(a)
            print("Graph adjacency matrix: ", graph_adj_mat)

            # Use DFS to find connected components
            visited = {}
            for key in graph_adj_mat.keys():
                visited[key] = False

            def dfs(node, visited, graph_adj_mat, connected_component):
                visited[node] = True
                connected_component.append(node)
                for neighbour in graph_adj_mat[node]:
                    if not visited[neighbour]:
                        dfs(neighbour, visited, graph_adj_mat, connected_component)
                return connected_component
            

            connected_components = []
            for key in graph_adj_mat.keys():
                if not visited[key]:
                    connected_component = []
                    connected_components.append(dfs(key, visited, graph_adj_mat, connected_component))
            print("Connected components: ", connected_components)

            # Visualize connected components
            connected_components_img = dataset.getImage(img_id)
            connected_components_img = torch.from_numpy(connected_components_img).permute(2, 0, 1).float()/ 255.0
            connected_components_img_np = connected_components_img.permute(1, 2, 0).cpu().numpy()

            def visualize_connected_components(img, kpts, connected_components):
                img = img.copy()
                # Rank based on the number of keypoints in the connected components
                connected_components = sorted(connected_components, key=lambda x: len(x), reverse=True)
                # Draw convex hulls around the connected components 



                for component in connected_components:
                    colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,0,255), (255,255,0), (255,255,255), (0,0,0), (200,200,200), (100,100,100), (150,150,150), (50,50,50)] + [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(100)]
                    for idx in component:
                        if idx >= len(colors):
                            colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
                        cv2.circle(img, (int(kpts[idx][0]), int(kpts[idx][1])), 3, colors[connected_components.index(component)], -1)

                    convex_hull = cv2.convexHull(np.array([kpts[idx] for idx in component]).astype(np.int32))

                    cv2.polylines(img, [convex_hull], True, colors[connected_components.index(component)], 2)



                return img
            
            connected_components_img_with_circles = visualize_connected_components(connected_components_img_np, kpts1_np, connected_components)
            cv2.imshow("Connected components", connected_components_img_with_circles)
            cv2.waitKey(2)

            # Only show the N components which has the highest average difference in lengths and have less than 10% of the keypoints
            # Find average difference in lengths for each connected component
            avg_diffs = []
            for component in connected_components:
                diffs = []
                for i in range(len(component)):
                    a,b = common_pairs[i]
                    diffs.append(np.linalg.norm(kpts1_np[a] - kpts1_np[b]))
                avg_diffs.append(np.mean(diffs))

            # Sort the connected components based on the average difference in lengths
            connected_components = [x for _, x in sorted(zip(avg_diffs, connected_components), reverse=True)]

            # Only show the N components which has the highest average difference in lengths and have less than 10% of the keypoints
            N = 3
            threshold = 0.1
            selected_components = []
            for component in connected_components:
                if len(component) < threshold * len(kpts1_np) and len(component) > 2: ## 3 because any reasonable dynamic object will have more than 3 keypoints
                    selected_components.append(component)
                if len(selected_components) == N:
                    break

            # Visualize selected components
            selected_components_img = dataset.getImage(img_id)
            
            selected_components_img = torch.from_numpy(selected_components_img).permute(2, 0, 1).float()/ 255.0

            selected_components_img_np = selected_components_img.permute(1, 2, 0).cpu().numpy()
            selected_components_img_with_circles = visualize_connected_components(selected_components_img_np, kpts1_np, selected_components)
            cv2.imshow("Selected components", selected_components_img_with_circles)
            cv2.waitKey(2)


            # SAVE PATH - Record a video object for each image shown using cv2.imshow
            

            
 



                        
            

            
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


        
        # if img_id > starting_img_id + 5: 
        #     # Create visualizer
        #     vis = o3d.visualization.Visualizer()
        #     vis.create_window()

        #     # Create a single point cloud from all frames
        #     global_map = o3d.geometry.PointCloud()
        #     all_points = []
        #     all_colors = []

        #     # Transform and combine all point clouds
        #     for frame_id, points in accumulated_clouds.items():
        #         if frame_id in skipped_frames or frame_id in invalid_frames:
        #             continue
        #         if frame_id not in global_poses:
        #             continue
        #         if np.array_equal(global_poses[frame_id], np.eye(4)):  # Skip identity poses
        #             continue
                    
        #         # Get points and colors
        #         pts = points.points
        #         colors = points.colors
                
        #         # Convert points to homogeneous coordinates
        #         pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))
                
        #         # Transform points using global pose
        #         transformed_pts = (global_poses[frame_id] @ pts_homog.T).T[:, :3]
                
        #         all_points.append(transformed_pts)
        #         all_colors.append(colors)

        #     # Combine all points into single point cloud
        #     if all_points:
        #         global_map.points = o3d.utility.Vector3dVector(np.vstack(all_points))
        #         global_map.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
                
        #         # Optional: Downsample to reduce density
        #         global_map = global_map.voxel_down_sample(voxel_size=0.1)
                
        #         # Add point cloud to visualizer
        #         vis.add_geometry(global_map)

        #         # Add camera poses as coordinate frames
        #         for frame_id, pose in global_poses.items():
        #             cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #             cam_frame.transform(pose)
        #             vis.add_geometry(cam_frame)

        #         # Add trajectory visualization
        #         points = []
        #         for pose in global_poses.values():
        #             points.append(pose[:3, 3])

        #         if points:
        #             # Create trajectory line set
        #             trajectory = o3d.geometry.LineSet()
        #             trajectory.points = o3d.utility.Vector3dVector(points)
        #             trajectory.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(points)-1)])
        #             trajectory.paint_uniform_color([1, 0, 0])  # Red color
        #             vis.add_geometry(trajectory)

        #         # Add world coordinate frame
        #         world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        #         vis.add_geometry(world_frame)

        #         # Set view control
        #         vis.get_view_control().set_zoom(0.5)
        #         vis.get_view_control().set_front([0, 0, -1])
        #         vis.get_view_control().set_up([0, -1, 0])

        #         # Run visualizer
        #         vis.run()
        #         vis.destroy_window()
        #     break

        img_id += 1