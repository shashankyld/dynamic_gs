import numpy as np
from scipy.spatial import Delaunay, KDTree
import cv2
import networkx as nx
import torch
import open3d as o3d
from typing import List, Dict, Set
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader
from core.frame import Frame
from utilities.utils_draw import draw_torch_image
from utilities.utils_depth import depth2pointcloud, depth2pcd


# Plot histogram of edge lengths in an image 
def hist_img(hist, bins, width=800, height=600):
    hist = cv2.normalize(hist, None, 0, 255, cv2.NORM_MINMAX)
    hist_img = np.zeros((height, width), dtype=np.uint8)
    bin_width = width // len(bins)
    for i, h in enumerate(hist):
        cv2.rectangle(hist_img, (i*bin_width, height), ((i+1)*bin_width, height - int(h)), 255, -1)
    # ADD TEXT TO SHOW NUMBER OF EDGES
    cv2.putText(hist_img, f'Number of Edges: {len(hist)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    # Add Axis and grid and names of axis
    cv2.line(hist_img, (0, height), (width, height), (255, 255, 255), 2)
    cv2.line(hist_img, (0, height), (0, 0), (255, 255, 255), 2)
    for i in range(1, 10):
        cv2.line(hist_img, (i*bin_width, height), (i*bin_width, 0), (255, 255, 255), 1)
    cv2.putText(hist_img, 'Edge Lengths', (width//2 - 50, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(hist_img, 'Number of Edges', (10, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    return hist_img

def delaunay_triangulation(frame):
    """
    Create delaunay triangulation from frame keypoints
    Args:
        frame: Frame object containing keypoints
    Returns:
        G: networkx Graph object containing Delaunay triangulation
    """
    # Get keypoints and convert to numpy if needed
    keypoints = frame.keypoints
    if isinstance(keypoints, torch.Tensor):
        keypoints_np = keypoints.cpu().numpy()
    else:
        keypoints_np = np.array(keypoints)

    # Create delaunay triangulation
    try:
        tri = Delaunay(keypoints_np)
    except Exception as e:
        print(f"Error in Delaunay triangulation: {e}")
        print(f"Keypoints shape: {keypoints_np.shape}")
        return None

    # Get simplices and create edges
    simplices = tri.simplices
    edges = np.vstack((simplices[:, [0, 1]],
                      simplices[:, [1, 2]],
                      simplices[:, [2, 0]]))

    # Create and return graph
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def draw_delaunay_triangulation(G, frame):
    """
    Draw delaunay triangulation on frame
    Args:
        G: networkx Graph object containing Delaunay triangulation
        frame: Frame object containing image and keypoints
    Returns:
        img: Image with drawn Delaunay triangulation
    """
    if G is None:
        print("No graph to draw")
        return None

    # Get keypoints and convert to numpy if needed
    keypoints = frame.keypoints
    if isinstance(keypoints, torch.Tensor):
        keypoints_np = keypoints.cpu().numpy()
    else:
        keypoints_np = np.array(keypoints)

    # Get image and convert to numpy if needed
    img = frame.image if hasattr(frame, 'image') else frame.img
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
    else:
        img_np = np.array(img)

    # Create visualization image
    vis_img = img_np.copy() if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Draw keypoints
    for keypoint in keypoints_np:
        cv2.circle(vis_img, 
                  (int(keypoint[0]), int(keypoint[1])), 
                  3, (0, 255, 0), -1)

    # Draw edges
    for edge in G.edges():
        pt1 = tuple(map(int, keypoints_np[edge[0]]))
        pt2 = tuple(map(int, keypoints_np[edge[1]]))
        cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)

    # Show image
    cv2.imshow("Delaunay Triangulation", vis_img)
    cv2.waitKey(1)
    
    return vis_img

def draw_delaunay_triangulation_using_G_kps(G, frame):
    """
    Draw delaunay triangulation on frame using keypoints from graph
    Args:
        G: networkx Graph object containing Delaunay triangulation
        frame: Frame object containing image and keypoints
    Returns:
        img: Image with drawn Delaunay triangulation
    """
    # Almost same except - only draw keypoints from graph
    if G is None:
        print("No graph to draw")
        return None
    
    # Get keypoints and convert to numpy if needed
    keypoints = frame.keypoints
    if isinstance(keypoints, torch.Tensor):
        keypoints_np = keypoints.cpu().numpy()
    else:
        keypoints_np = np.array(keypoints)

    # Get image and convert to numpy if needed
    img = frame.image if hasattr(frame, 'image') else frame.img
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
    else:
        img_np = np.array(img)

    # Create visualization image
    vis_img = img_np.copy() if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Draw only keypoints that are in the graph
    for node in G.nodes():
        if node < len(keypoints_np):
            keypoint = keypoints_np[node]
            cv2.circle(vis_img, 
                      (int(keypoint[0]), int(keypoint[1])), 
                      3, (0, 255, 0), -1)

    # Draw edges
    for edge in G.edges():
        if edge[0] < len(keypoints_np) and edge[1] < len(keypoints_np):
            pt1 = tuple(map(int, keypoints_np[edge[0]]))
            pt2 = tuple(map(int, keypoints_np[edge[1]]))
            cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)

    # Show image
    cv2.imshow("Delaunay Triangulation (Graph KPs Only)", vis_img)
    cv2.waitKey(1)
    
    return vis_img


def matches_with_last_dealunay(frame, frame_with_last_dealunay):
    """ 
    Everytime delaunay triangulation is performed, we need to add it to slam object.
    This happens for every keyframe (Check implimentation in slam.py)
    Match kps between frames, and for every edge in the prev_delaunay that has vertices that are common in both frames, add it to the current delaunay
    Args:
        frame: Frame object containing keypoints
        frame_with_last_dealunay: Frame object containing keypoints and delaunay
    Returns:
        G: networkx Graph object containing Delaunay triangulation

    """
    # Get keypoints and convert to numpy if needed
    keypoints = frame.keypoints
    if isinstance(keypoints, torch.Tensor):
        keypoints_np = keypoints.cpu().numpy()
    else:
        keypoints_np = np.array(keypoints)

    # Get keypoints and convert to numpy if needed
    keypoints_last = frame_with_last_dealunay.keypoints
    if isinstance(keypoints_last, torch.Tensor):
        keypoints_last_np = keypoints_last.cpu().numpy()
    else:
        keypoints_last_np = np.array(keypoints_last)

    # Get delaunay graph from last frame
    G_last = frame_with_last_dealunay.delaunay

    # Create KDTree for fast nearest neighbor search
    kdtree = KDTree(keypoints_np)

    # Find matches between frames
    matches = []
    for i, keypoint in enumerate(keypoints_last_np):
        _, idx = kdtree.query(keypoint)
        matches.append((i, idx))

    # Create new graph
    G = nx.Graph()
    G.add_edges_from(G_last.edges())

    # Add matches to graph
    for match in matches:
        if match[0] in G_last and match[1] in G_last:
            G.add_edge(match[0], match[1])

    return G

def draw_matches_with_last_dealunay(G, frame, frame_with_last_dealunay):
    """
    Draw matches between current frame and frame with last delaunay
    Args:
        G: networkx Graph object containing matches between frames
        frame: Frame object containing image and keypoints
        frame_with_last_dealunay: Frame object containing image, keypoints, and delaunay from last frame
    Returns:
        img: Image with drawn matches
    """
    if G is None:
        print("No graph to draw")
        return None

    # Get keypoints and convert to numpy if needed
    keypoints = frame.keypoints
    if isinstance(keypoints, torch.Tensor):
        keypoints_np = keypoints.cpu().numpy()
    else:
        keypoints_np = np.array(keypoints)

    # Get keypoints and convert to numpy if needed
    keypoints_last = frame_with_last_dealunay.keypoints
    if isinstance(keypoints_last, torch.Tensor):
        keypoints_last_np = keypoints_last.cpu().numpy()
    else:
        keypoints_last_np = np.array(keypoints_last)

    # Get image and convert to numpy if needed
    img = frame.image if hasattr(frame, 'image') else frame.img
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
    else:
        img_np = np.array(img)

    # Create visualization image
    vis_img = img_np.copy() if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # Draw keypoints
    for keypoint in keypoints_np:
        cv2.circle(vis_img, 
                  (int(keypoint[0]), int(keypoint[1])), 
                  3, (0, 255, 0), -1)

    # Draw matches
    for edge in G.edges():
        pt1 = tuple(map(int, keypoints_last_np[edge[0]]))
        pt2 = tuple(map(int, keypoints_np[edge[1]]))
        cv2.line(vis_img, pt1, pt2, (255, 0, 0), 1)

    # Show image
    cv2.imshow("Matches with last delaunay", vis_img)
    cv2.waitKey(1)
    
    return vis_img

def matches_with_k_frames_away_with_prev_delaunay_edges(frame, slam, k, compare_frame=None):
    """Find and visualize common keypoints matched across current frame, last keyframe, and k-frames-away frame.
    
    Args:
        frame: Current frame
        slam: SLAM system containing previous frames and matcher
        k: Number of frames to look back
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - matches_curr_prev: Matches between current frame and last keyframe (curr_idx, prev_idx)
            - matches_curr_k: Matches between current frame and k-frames-away frame (curr_idx, k_idx)
            - matches_prev_k: Matches between last keyframe and k-frames-away frame (prev_idx, k_idx)
            - common_matches: Array of common matches across all three frames 
                             (curr_idx, prev_idx, k_idx)
    """
    # Get the frames
    frame = frame
    try:
        frame_with_last_delaunay = slam.map.get_last_keyframe()
    except IndexError:
        print(slam.map.keyframes)
        print("No keyframes in map")
        return None, None, None, None
    if compare_frame is None:
        frame_k_frames_away = get_frame_from_pyslam_dataloader(slam.dataset, slam.groundtruth, frame.id - k, slam.config)
    else:
        frame_k_frames_away = compare_frame
    # Get matches between frames using the tracker
    matches_curr_prev = slam.tracker.match_frames(frame, frame_with_last_delaunay)
    matches_curr_k = slam.tracker.match_frames(frame, frame_k_frames_away)
    matches_prev_k = slam.tracker.match_frames(frame_with_last_delaunay, frame_k_frames_away)

    # Create visualization image
    h1, w1 = frame.image.shape[:2]
    h2, w2 = frame_with_last_delaunay.image.shape[:2]
    h3, w3 = frame_k_frames_away.image.shape[:2]
    
    # Create empty canvas with maximum height and sum of widths
    max_h = max(h1, h2, h3)
    vis_img = np.zeros((max_h, w1 + w2 + w3, 3), dtype=np.uint8)
    
    # Convert images to BGR if they're not already
    def ensure_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    # Place images
    img1 = ensure_bgr(frame.image)
    img2 = ensure_bgr(frame_with_last_delaunay.image)
    img3 = ensure_bgr(frame_k_frames_away.image)
    
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2
    vis_img[:h3, w1+w2:] = img3

    # Define green color for common points
    GREEN = (0, 255, 0)

    # Helper function to draw keypoint
    def draw_keypoint(img, kp, color, offset_x=0):
        cv2.circle(img, 
                  (int(kp[0]) + offset_x, int(kp[1])), 
                  3, color, -1)

    # Find common keypoints (present in all three match sets)
    # Create dictionaries for quick lookup
    curr_prev_dict = dict(matches_curr_prev)  # curr_idx -> prev_idx
    curr_k_dict = dict(matches_curr_k)        # curr_idx -> k_idx
    prev_k_dict = dict(matches_prev_k)        # prev_idx -> k_idx

    # Find keypoints common across all three frames
    common_matches = []
    for curr_idx, prev_idx in curr_prev_dict.items():
        if curr_idx in curr_k_dict:  # Matches with k-frame
            k_idx = curr_k_dict[curr_idx]
            if prev_idx in prev_k_dict and prev_k_dict[prev_idx] == k_idx:  # Consistent triangle
                common_matches.append([curr_idx, prev_idx, k_idx])

    common_matches = np.array(common_matches, dtype=np.int32)

    # Draw only common keypoints
    # Current frame (left)
    for i, kp in enumerate(frame.keypoints):
        if i in common_matches[:, 0]:
            draw_keypoint(vis_img, kp, GREEN)

    # Previous keyframe (middle)
    for i, kp in enumerate(frame_with_last_delaunay.keypoints):
        if i in common_matches[:, 1]:
            draw_keypoint(vis_img, kp, GREEN, w1)

    # K-frames-away frame (right)
    for i, kp in enumerate(frame_k_frames_away.keypoints):
        if i in common_matches[:, 2]:
            draw_keypoint(vis_img, kp, GREEN, w1+w2)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f'Current Frame (id:{frame.id})', (10, 30), font, 1, (255,0,0), 2)
    cv2.putText(vis_img, f'Last Keyframe (id:{frame_with_last_delaunay.id})', (w1+10, 30), font, 1, (255,0,0), 2)
    cv2.putText(vis_img, f'K-Frame Away (id:{frame_k_frames_away.id})', (w1+w2+10, 30), font, 1, (255,0,0), 2)

    ## ADD TEXT TO SHOW NUMBER OF COMMON POINTS
    cv2.putText(vis_img, f'Common Points: {len(common_matches)}', (10, 60), font, 1, (0,0,255), 2)
    

    # Show visualization
    cv2.imshow("Common Keypoints Visualization", vis_img)
    cv2.waitKey(1)

    # Return all matches and common matches
    return matches_curr_prev, matches_curr_k, matches_prev_k, common_matches


def G_all_frames(curr_frame, slam):
    """Create and visualize Delaunay graphs for current frame, previous keyframe, and comparison frame.
    
    Args:
        curr_frame: Current frame being processed
        slam: SLAM system containing previous frames and configuration
        
    Returns:
        tuple: (G_curr, G_prev, G_compare) - Delaunay graphs for each frame
    """
    curr_frame = curr_frame
    prev_delaunay_frame = slam.map.get_last_keyframe()
    print("Prev Delaunay Frame: ", prev_delaunay_frame.id)
    compare_frame = get_frame_from_pyslam_dataloader(
        slam.dataset, slam.groundtruth, 
        curr_frame.id - slam.config.NumFramesAway, 
        slam.config
    )
    
    # Get matches between all three frames
    matches_curr_prev, matches_curr_k, matches_prev_k, common_matches = \
        matches_with_k_frames_away_with_prev_delaunay_edges(
            curr_frame, slam, slam.config.NumFramesAway, compare_frame= compare_frame
        )
    
    if common_matches is None or len(common_matches) < 3:
        print("Not enough common matches for triangulation")
        return None, None, None
        
    '''
    # Get existing Delaunay graph from keyframe
    G_prev = prev_delaunay_frame._delaunay # This will inevitably have missing edges and trick the system into thinking there are moving components.
    '''

    # Create a new Delaunay graph for the keyframe from the common matches [1] by force 
    fake_kf = Frame(frame_id=prev_delaunay_frame.id, timestamp=prev_delaunay_frame.timestamp)
    fake_kf._image = prev_delaunay_frame.image
    fake_kf.keypoints = prev_delaunay_frame.keypoints[common_matches[:, 1]]
    
    G_prev = delaunay_triangulation(fake_kf)
    draw_delaunay_triangulation(G_prev, fake_kf)

    # Now with certainity, every edge in G_prev exists in other two frames as well.
    # Create new Delaunay graph for current frame, comparison frame - using the matches 
    G_curr = nx.Graph()
    G_compare = nx.Graph()
    
    # Create a mapping from prev indices (in common_matches) to current and comparison indices
    prev_to_curr = {prev_idx: curr_idx for curr_idx, prev_idx, _ in common_matches}
    prev_to_compare = {prev_idx: k_idx for _, prev_idx, k_idx in common_matches}
    
    # Add all nodes first
    for node in G_prev.nodes():
        # Map the node index from keyframe to current and comparison frames
        curr_node = prev_to_curr[common_matches[node, 1]]
        compare_node = prev_to_compare[common_matches[node, 1]]
        
        G_curr.add_node(curr_node)
        G_compare.add_node(compare_node)
    
    # Add all edges
    for edge in G_prev.edges():
        i, j = edge
        
        # Map the edge indices from keyframe to current and comparison frames
        curr_i = prev_to_curr[common_matches[i, 1]]
        curr_j = prev_to_curr[common_matches[j, 1]]
        
        compare_i = prev_to_compare[common_matches[i, 1]]
        compare_j = prev_to_compare[common_matches[j, 1]]
        
        # Add edges to respective graphs
        G_curr.add_edge(curr_i, curr_j)
        G_compare.add_edge(compare_i, compare_j)
    
    # Create visualization frames
    fake_curr = Frame(frame_id=curr_frame.id, timestamp=curr_frame.timestamp)
    fake_curr._image = curr_frame.image
    fake_curr.keypoints = curr_frame.keypoints
    
    fake_compare = Frame(frame_id=compare_frame.id, timestamp=compare_frame.timestamp)
    fake_compare._image = compare_frame.image
    fake_compare.keypoints = compare_frame.keypoints

    # Visualize all graphs
    curr_img = draw_delaunay_triangulation_using_G_kps(G_curr, fake_curr)
    compare_img = draw_delaunay_triangulation_using_G_kps(G_compare, fake_compare)
    
    # Create a combined visualization
    h1, w1 = curr_img.shape[:2]
    h2, w2 = fake_kf._image.shape[:2]
    h3, w3 = compare_img.shape[:2]
    
    # Create empty canvas with maximum height and sum of widths
    max_h = max(h1, h2, h3)
    vis_img = np.zeros((max_h, w1 + w2 + w3, 3), dtype=np.uint8)
    
    # Add images to visualization
    vis_img[:h1, :w1] = curr_img
    vis_img[:h2, w1:w1+w2] = draw_delaunay_triangulation(G_prev, fake_kf)
    vis_img[:h3, w1+w2:] = compare_img
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f'Current Frame (id:{curr_frame.id})', (10, 30), font, 0.8, (255,0,0), 2)
    cv2.putText(vis_img, f'Keyframe (id:{prev_delaunay_frame.id})', (w1+10, 30), font, 0.8, (255,0,0), 2)
    cv2.putText(vis_img, f'Compare Frame (id:{compare_frame.id})', (w1+w2+10, 30), font, 0.8, (255,0,0), 2)
    
    # Show combined visualization
    cv2.imshow("Delaunay Graphs Visualization", vis_img)
    cv2.waitKey(1)
    
    return G_curr, G_prev, G_compare


def get_static_dynamic_edges(curr_frame, slam):
    """
    Identify static and dynamic edges by comparing edge lengths in 3D between frames.
    Remove dynamic edges from the current frame's graph.
    
    Args:
        curr_frame: Current frame being processed
        slam: SLAM system containing previous frames and configuration
        
    Returns:
        tuple: (G_curr_static, dynamic_edges) - Graph with dynamic edges removed and list of dynamic edges
    """
    curr_frame = curr_frame
    prev_delaunay_frame = slam.map.get_last_keyframe()
    print("Prev Delaunay Frame: ", prev_delaunay_frame.id)
    compare_frame = get_frame_from_pyslam_dataloader(
        slam.dataset, slam.groundtruth, 
        curr_frame.id - slam.config.NumFramesAway,  # Use the parameter from SlamParameters
        slam.config
    )
    
    # Get matches between all three frames
    matches_curr_prev, matches_curr_k, matches_prev_k, common_matches = \
        matches_with_k_frames_away_with_prev_delaunay_edges(
            curr_frame, slam, slam.config.NumFramesAway, compare_frame=compare_frame
        )
    
    if common_matches is None or len(common_matches) < 3:
        print("Not enough common matches for triangulation")
        return None, None
        
    #
    # For each idx in curr, prev, compare, if the depth is invalid in their respective frames, remove them from the common_matches
    curr_depth = curr_frame._depth
    prev_depth = prev_delaunay_frame._depth
    compare_depth = compare_frame._depth


    def check_depth(frame, idx, depth):
        kp = frame.keypoints[idx]
        x, y = int(kp[0]), int(kp[1])
        if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
            return depth[y, x] == 0
        return True
    
    common_matches_with_good_depth = []
    for idx in range(len(common_matches)):
        curr_idx = common_matches[idx, 0]
        prev_idx = common_matches[idx, 1]
        compare_idx = common_matches[idx, 2]
        if check_depth(curr_frame, curr_idx, curr_depth) or check_depth(prev_delaunay_frame, prev_idx, prev_depth) or check_depth(compare_frame, compare_idx, compare_depth):
            continue
        common_matches_with_good_depth.append([curr_idx, prev_idx, compare_idx])

    common_matches= np.array(common_matches_with_good_depth, dtype=np.int32)
    print("Common Matches with good depth: ", common_matches_with_good_depth)




    print("Common Matches: ", common_matches)
    curr_kp_idxs = common_matches[:, 0]
    prev_kp_idxs = common_matches[:, 1]
    compare_kp_idxs = common_matches[:, 2]


    
    camera_matrix = curr_frame.camera_matrix
    # Fake Prev_KF 
    fake_kf = Frame(frame_id=prev_delaunay_frame.id, timestamp=prev_delaunay_frame.timestamp, camera_matrix=camera_matrix)
    fake_kf._image = prev_delaunay_frame.image
    fake_kf._depth = prev_delaunay_frame.depth
    fake_kf.keypoints = prev_delaunay_frame.keypoints[prev_kp_idxs]

    # Fake Current Frame
    fake_curr = Frame(frame_id=curr_frame.id, timestamp=curr_frame.timestamp, camera_matrix=camera_matrix)
    fake_curr._image = curr_frame.image
    fake_curr._depth = curr_frame.depth
    fake_curr.keypoints = curr_frame.keypoints[curr_kp_idxs]
    

    # Fake Compare Frame
    fake_compare = Frame(frame_id=compare_frame.id, timestamp=compare_frame.timestamp  , camera_matrix=camera_matrix)
    fake_compare._image = compare_frame.image
    fake_compare._depth = compare_frame.depth
    fake_compare.keypoints = compare_frame.keypoints[compare_kp_idxs]

    # Create Delaunay graph for the keyframe from the common matches
    G_prev = delaunay_triangulation(fake_kf)
    
    draw_delaunay_triangulation_using_G_kps(G_prev, fake_kf)

    edges_list = list(G_prev.edges())
    print("Edges List: ", edges_list)   

    prev_kps_3d = fake_kf.get_3d_kps()
    curr_kps_3d = fake_curr.get_3d_kps()
    compare_kps_3d = fake_compare.get_3d_kps()

    # Calculate edge lengths in 3D
    for edge in edges_list:
        i, j = edge
        print("EDGE COORDINATES: ", prev_kps_3d[i], prev_kps_3d[j])
        prev_edge_len = np.linalg.norm(prev_kps_3d[j] - prev_kps_3d[i])
        curr_edge_len = np.linalg.norm(curr_kps_3d[j] - curr_kps_3d[i])
        compare_edge_len = np.linalg.norm(compare_kps_3d[j] - compare_kps_3d[i])

        print(f"Edge lengths: Prev: {prev_edge_len}, Curr: {curr_edge_len}, Compare: {compare_edge_len}")

    

    # DEPTH IMAGE VISUALIZTION 
    # Show white image where depth is available and black where depth is not available or zero and also show kps in the image 
    # Also show how many kps are there in the image with zero depth and non-zero depth
    # Only for the current frame

    # Create a binary mask for valid depth
    depth_valid = (fake_curr._depth > 0).astype(np.uint8) * 255

    # Convert to 3-channel for visualization
    depth_vis = cv2.cvtColor(depth_valid, cv2.COLOR_GRAY2BGR)

    zero_depth_count = 0
    non_zero_depth_count = 0

    # Ensure we have keypoints
    if fake_curr.keypoints is not None:
        for kp in fake_curr.keypoints:
            x, y = int(kp[0]), int(kp[1])
            # Check image bounds
            if 0 <= x < fake_curr._depth.shape[1] and 0 <= y < fake_curr._depth.shape[0]:
                if fake_curr._depth[y, x] > 0:
                    non_zero_depth_count += 1
                    color = (0, 255, 0)  # Green circle for valid depth
                else:
                    zero_depth_count += 1
                    color = (0, 0, 255)  # Red circle for zero depth
                cv2.circle(depth_vis, (x, y), 3, color, -1)

    # Add text overlay
    info_text = f"Zero Depth KPs: {zero_depth_count}, Non-zero Depth KPs: {non_zero_depth_count}"
    cv2.putText(depth_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show visualization
    cv2.imshow("Depth Visualization", depth_vis)
    cv2.waitKey(1)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    full_pc_pts, full_pc_colors = fake_curr.get_points()
    full_pc = o3d.geometry.PointCloud()
    full_pc.points = o3d.utility.Vector3dVector(full_pc_pts)
    full_pc.colors = o3d.utility.Vector3dVector(full_pc_colors)
    
    kps_pcd = o3d.geometry.PointCloud()
    kps_pcd.points = o3d.utility.Vector3dVector(curr_kps_3d)
    kps_pcd.colors = o3d.utility.Vector3dVector(np.random.rand(len(curr_kps_3d), 3))

    edges = []
    for edge in edges_list:
        i, j = edge
        # Create line objects in open3d
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([curr_kps_3d[i], curr_kps_3d[j]])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([np.random.rand(3)])
        edges.append(line)

    # # Visualize point cloud and edges
    o3d.visualization.draw_geometries([kps_pcd] + edges + [full_pc] + [axes])

                 
