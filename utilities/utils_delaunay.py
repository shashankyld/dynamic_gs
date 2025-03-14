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

def draw_delaunay_triangulation_using_G_kps(G, frame, title=""):
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
    cv2.imshow("Delaunay Triangulation (Graph KPs Only)" + title, vis_img)
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

def draw_static_dynamic_edges(G_static, G_dynamic, frame):
    """
    # Static in green and dynamic in red
    """
    if G_static is None or G_dynamic is None:
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

    # Draw static edges in green
    for edge in G_static.edges():
        pt1 = tuple(map(int, keypoints_np[edge[0]]))
        pt2 = tuple(map(int, keypoints_np[edge[1]]))
        cv2.line(vis_img, pt1, pt2, (0, 255, 0), 1)

    # Draw dynamic edges in red
    for edge in G_dynamic.edges():
        pt1 = tuple(map(int, keypoints_np[edge[0]]))
        pt2 = tuple(map(int, keypoints_np[edge[1]]))
        cv2.line(vis_img, pt1, pt2, (0, 0, 255), 1)

    # Show image
    cv2.imshow("Static and Dynamic Edges", vis_img)
    cv2.waitKey(1)
    
    return vis_img


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
    
    # draw_delaunay_triangulation_using_G_kps(G_prev, fake_kf)

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

    
    # Create a new graph for the current frame 
    G_curr = nx.Graph()
    G_curr_dynamic_edges = nx.Graph()
    # Add edges to the graph when the length of the edge is within a threshold compared to the compare frame and prev frame
    for edge in edges_list:
        i, j = edge
        prev_edge_len = np.linalg.norm(prev_kps_3d[j] - prev_kps_3d[i])
        curr_edge_len = np.linalg.norm(curr_kps_3d[j] - curr_kps_3d[i])
        compare_edge_len = np.linalg.norm(compare_kps_3d[j] - compare_kps_3d[i])
        if abs(curr_edge_len - compare_edge_len) < slam.config.DYNAMIC_EDGE_THRESHOLD and abs(curr_edge_len - prev_edge_len) < slam.config.DYNAMIC_EDGE_THRESHOLD :
            G_curr.add_edge(i, j)
        else:
            G_curr_dynamic_edges.add_edge(i, j)

    # Visualize the graph
    draw_delaunay_triangulation_using_G_kps(G_curr, fake_curr, title = "Static Edges")
    draw_delaunay_triangulation_using_G_kps(G_curr_dynamic_edges, fake_curr, title = "Dynamic Edges")

    draw_static_dynamic_edges(G_curr, G_curr_dynamic_edges, fake_curr)


def get_static_dynamic_edges_batch(curr_frame, slam, batch_size = 5, stride = 2):
    """
    Instead of just comparing the last-keyframe and k-frames-away frame, 
    we compare the current frame edges with a batch of frames over time for better temporal consistency.
    
    Key Insight:
    - If an edge length in 3D is static, it should remain nearly the same across multiple frames.
    - We collect the 3D lengths of each edge over the batch, apply median filtering to reject outliers, 
      and then compute the variance. If the variance is high, we mark the edge as dynamic.
    
    Args:
        curr_frame: Current frame being processed.
        slam: SLAM system containing previous frames and configuration.
        batch_size: How many frames to consider for comparison.
        stride: Step size between frames when collecting the batch.

    Returns:
        (G_curr_static, G_curr_dynamic): A tuple of two graphs:
            - G_curr_static: Graph containing only static edges.
            - G_curr_dynamic: Graph containing edges marked as dynamic.
    """
    import numpy as np
    import networkx as nx
    from scipy.stats import median_abs_deviation

    # Grab the last keyframe for reference
    prev_delaunay_frame = slam.map.get_last_keyframe()
    
    # Create a "batch" of frames to compare over time
    # We'll collect frames from (current_id - stride * i) in [1..batch_size]
    curr_id = curr_frame.id
    frames_batch = []
    for i in range(batch_size):
        try:
            frame_id = curr_id - (i+1)*stride
            if frame_id < 0:
                break
            f = get_frame_from_pyslam_dataloader(slam.dataset, slam.groundtruth, frame_id, slam.config)
        except KeyError:
            break
        if f is not None:
            frames_batch.append(f)
    # frames_batch now holds up to batch_size older frames

    if not frames_batch:
        print("No older frames found for temporal consistency. Reverting to single-frame check.")
        # Fallback: just do a quick check with the last keyframe
        return get_static_dynamic_edges(curr_frame, slam)
    
    # STEP 1: Match common keypoints across all frames
    print(f"Finding common keypoints across {len(frames_batch) + 2} frames...")
    
    # First match keypoints between current frame and previous keyframe
    matches_curr_prev = slam.tracker.match_frames(curr_frame, prev_delaunay_frame)
    if len(matches_curr_prev) < 10:
        print("Not enough matches between current frame and keyframe. Reverting to single-frame check.")
        return get_static_dynamic_edges(curr_frame, slam)
    
    # Create mappings between current and keyframe indices
    curr_to_prev = {curr_idx: prev_idx for curr_idx, prev_idx in matches_curr_prev}
    prev_to_curr = {prev_idx: curr_idx for curr_idx, prev_idx in matches_curr_prev}
    
    # Now match keypoints between each batch frame and the keyframe
    batch_matches = []
    for f in frames_batch:
        matches = slam.tracker.match_frames(f, prev_delaunay_frame)
        if len(matches) < 10:
            print(f"Not enough matches for batch frame {f.id}. Skipping.")
            continue
        batch_matches.append(matches)
    
    if not batch_matches:
        print("No batch frames with enough matches. Reverting to single-frame check.")
        return get_static_dynamic_edges(curr_frame, slam)
    
    # Find keypoints common to all frames (keyframe, current, and all batch frames)
    common_prev_indices = set(prev_to_curr.keys())
    for matches in batch_matches:
        batch_to_prev = {batch_idx: prev_idx for batch_idx, prev_idx in matches}
        common_prev_indices &= set(batch_to_prev.values())
    
    if len(common_prev_indices) < 10:
        print(f"Only {len(common_prev_indices)} keypoints common across all frames. Reverting to single-frame check.")
        return get_static_dynamic_edges(curr_frame, slam)
    
    print(f"Found {len(common_prev_indices)} keypoints common to all frames")
    
    # STEP 2: Filter out keypoints with invalid depth
    valid_prev_indices = []
    for idx in common_prev_indices:
        # Check depth in keyframe
        kp = prev_delaunay_frame.keypoints[idx]
        x, y = int(kp[0]), int(kp[1])
        if x < 0 or x >= prev_delaunay_frame.depth.shape[1] or y < 0 or y >= prev_delaunay_frame.depth.shape[0]:
            continue
        if prev_delaunay_frame.depth[y, x] <= 0:
            continue
        
        # Check depth in current frame
        curr_idx = prev_to_curr[idx]
        kp = curr_frame.keypoints[curr_idx]
        x, y = int(kp[0]), int(kp[1])
        if x < 0 or x >= curr_frame.depth.shape[1] or y < 0 or y >= curr_frame.depth.shape[0]:
            continue
        if curr_frame.depth[y, x] <= 0:
            continue
        
        # Check depth in all batch frames
        valid_in_all_batch = True
        for batch_idx, matches in enumerate(batch_matches):
            batch_to_prev = dict(matches)
            batch_to_prev_inv = {v: k for k, v in batch_to_prev.items()}
            if idx not in batch_to_prev_inv:
                valid_in_all_batch = False
                break
            
            batch_frame = frames_batch[batch_idx]
            batch_kp_idx = batch_to_prev_inv[idx]
            kp = batch_frame.keypoints[batch_kp_idx]
            x, y = int(kp[0]), int(kp[1])
            if x < 0 or x >= batch_frame.depth.shape[1] or y < 0 or y >= batch_frame.depth.shape[0]:
                valid_in_all_batch = False
                break
            if batch_frame.depth[y, x] <= 0:
                valid_in_all_batch = False
                break
        
        if valid_in_all_batch:
            valid_prev_indices.append(idx)
    
    if len(valid_prev_indices) < 10:
        print(f"Only {len(valid_prev_indices)} keypoints with valid depth across all frames. Reverting to single-frame check.")
        return get_static_dynamic_edges(curr_frame, slam)
    
    print(f"Found {len(valid_prev_indices)} keypoints with valid depth across all frames")
    
    # STEP 3: Create fake frames with only the matched keypoints
    camera_matrix = curr_frame.camera_matrix
    
    # Create fake keyframe
    fake_kf = Frame(
        frame_id=prev_delaunay_frame.id,
        timestamp=prev_delaunay_frame.timestamp,
        camera_matrix=camera_matrix
    )
    fake_kf._image = prev_delaunay_frame.image
    fake_kf._depth = prev_delaunay_frame.depth
    fake_kf.keypoints = prev_delaunay_frame.keypoints[valid_prev_indices]
    
    # Create mapping from original indices to new indices
    prev_old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_prev_indices)}
    
    # Create fake current frame
    fake_curr = Frame(
        frame_id=curr_frame.id,
        timestamp=curr_frame.timestamp,
        camera_matrix=camera_matrix
    )
    fake_curr._image = curr_frame.image
    fake_curr._depth = curr_frame.depth
    fake_curr.keypoints = curr_frame.keypoints[[prev_to_curr[idx] for idx in valid_prev_indices]]
    
    # Create fake batch frames
    fake_batch_frames = []
    for batch_idx, matches in enumerate(batch_matches):
        batch_frame = frames_batch[batch_idx]
        batch_to_prev = dict(matches)
        batch_to_prev_inv = {v: k for k, v in batch_to_prev.items()}
        
        fake_batch = Frame(
            frame_id=batch_frame.id,
            timestamp=batch_frame.timestamp,
            camera_matrix=camera_matrix
        )
        fake_batch._image = batch_frame.image
        fake_batch._depth = batch_frame.depth
        fake_batch.keypoints = batch_frame.keypoints[[batch_to_prev_inv[idx] for idx in valid_prev_indices]]
        fake_batch_frames.append(fake_batch)
    
    # STEP 4: Build Delaunay triangulation on the keyframe
    G_prev = delaunay_triangulation(fake_kf)
    if G_prev is None or len(G_prev.edges()) == 0:
        print("Failed to create Delaunay triangulation. Exiting.")
        return nx.Graph(), nx.Graph()
    
    edges_list = list(G_prev.edges())
    print(f"Delaunay triangulation has {len(edges_list)} edges")
    
    # Build new graphs for static and dynamic edges
    G_curr_static = nx.Graph()
    G_curr_dynamic = nx.Graph()
    
    # STEP 5 & 6: Compute edge lengths across frames, apply median filtering, and determine static/dynamic
    # Get 3D points for all frames
    kf_3d_pts = fake_kf.get_3d_kps()
    curr_3d_pts = fake_curr.get_3d_kps()
    batch_3d_pts = [f.get_3d_kps() for f in fake_batch_frames]
    
    # For visualization: collect edge lengths statistics
    all_edge_lengths = []
    all_edge_variances = []
    
    for edge in edges_list:
        i, j = edge
        
        # Get edge lengths across all frames
        kf_len = np.linalg.norm(kf_3d_pts[j] - kf_3d_pts[i])
        curr_len = np.linalg.norm(curr_3d_pts[j] - curr_3d_pts[i])
        
        batch_lens = []
        for pts_3d in batch_3d_pts:
            batch_len = np.linalg.norm(pts_3d[j] - pts_3d[i])
            batch_lens.append(batch_len)
        
        # Combine all lengths
        all_lens = [kf_len, curr_len] + batch_lens
        
        # STEP 5: Apply median filtering to reject outliers
        if len(all_lens) >= 3:  # Need at least 3 samples for meaningful filtering
            all_lens_array = np.array(all_lens)
            median_len = np.median(all_lens_array)
            mad = median_abs_deviation(all_lens_array)
            
            # Filter out lengths that are more than 3 MADs from the median
            mask = np.abs(all_lens_array - median_len) <= 3 * mad
            filtered_lens = all_lens_array[mask]
        else:
            filtered_lens = np.array(all_lens)
        
        if len(filtered_lens) < 2:
            # Not enough samples after filtering
            G_curr_dynamic.add_edge(i, j)
            continue
        
        # STEP 6: Calculate variance to determine static vs dynamic
        var_len = np.var(filtered_lens)
        mean_len = np.mean(filtered_lens)
        # Normalize variance by the mean length for scale invariance
        normalized_var = var_len / (mean_len**2 + 1e-6)  # Add small epsilon to avoid division by zero
        
        # For visualization
        all_edge_lengths.append(mean_len)
        all_edge_variances.append(normalized_var)
        
        # Get dynamic threshold from config or use default
        THRESH_VAR = slam.config.DYNAMIC_EDGE_THRESHOLD_VAR 
        
        if normalized_var < THRESH_VAR:
            G_curr_static.add_edge(i, j)
        else:
            G_curr_dynamic.add_edge(i, j)
    
    # STEP 7: Visualize results
    # Create a visualization showing static vs dynamic edges
    draw_static_dynamic_edges(G_curr_static, G_curr_dynamic, fake_curr)
    
    # Create histogram of edge lengths and variances
    if len(all_edge_lengths) > 0:
        lengths_hist = np.histogram(all_edge_lengths, bins=30)
        variances_hist = np.histogram(all_edge_variances, bins=30)
        
        lengths_img = hist_img(lengths_hist[0], lengths_hist[1])
        cv2.imshow("Edge Lengths Distribution", lengths_img)
        
        variances_img = hist_img(variances_hist[0], variances_hist[1])
        cv2.putText(variances_img, f'Threshold: {THRESH_VAR:.4f}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Edge Variances Distribution", variances_img)
        
        cv2.waitKey(1)
    
    # STEP 8: Return results
    return (G_curr_static, G_curr_dynamic)
