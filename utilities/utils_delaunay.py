import numpy as np
from scipy.spatial import Delaunay, KDTree
import cv2
import networkx as nx
import torch
import open3d as o3d
from typing import List, Dict, Set
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader

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

    # Helper function to draw keypoints
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
        
    # Get existing Delaunay graph from keyframe
    G_prev = prev_delaunay_frame._delaunay
    print("Prev Delaunay Graph: ", G_prev)
    if G_prev is None:
        print("No Delaunay graph in keyframe")
        return None, None, None

    # Create mappings for common points
    curr_to_prev = dict(zip(common_matches[:, 0], common_matches[:, 1]))
    curr_to_k = dict(zip(common_matches[:, 0], common_matches[:, 2]))
    prev_to_k = dict(zip(common_matches[:, 1], common_matches[:, 2]))
    
    # Create inverse mappings for easier lookup
    prev_to_curr = {v: k for k, v in curr_to_prev.items()}
    k_to_curr = {v: k for k, v in curr_to_k.items()}
    k_to_prev = {v: k for k, v in prev_to_k.items()}
    
    # Create new graphs for current and comparison frames
    G_curr = nx.Graph()
    G_compare = nx.Graph()
    
    # Transfer edges from previous frame's Delaunay only if valid mappings exist
    for edge in G_prev.edges():
        v1, v2 = edge
        
        # Add edge to comparison frame if both vertices have mappings
        if v1 in prev_to_k and v2 in prev_to_k:
            k1, k2 = prev_to_k[v1], prev_to_k[v2]
            G_compare.add_edge(k1, k2)
            
        # Add edge to current frame if both vertices have mappings
        if v1 in prev_to_curr and v2 in prev_to_curr:
            c1, c2 = prev_to_curr[v1], prev_to_curr[v2]
            G_curr.add_edge(c1, c2)

    # If either graph is empty, consider triangulation failed
    if len(G_curr.edges()) == 0 or len(G_compare.edges()) == 0:
        print(f"Warning: Empty graph generated. G_curr: {len(G_curr.edges())} edges, G_compare: {len(G_compare.edges())} edges")
        return None, None, None
    
    # Create visualization
    h1, w1 = curr_frame.image.shape[:2]
    h2, w2 = prev_delaunay_frame.image.shape[:2]
    h3, w3 = compare_frame.image.shape[:2]
    
    max_h = max(h1, h2, h3)
    vis_img = np.zeros((max_h, w1 + w2 + w3, 3), dtype=np.uint8)
    
    # Convert images to BGR
    def ensure_bgr(img):
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img
    
    img1 = ensure_bgr(curr_frame.image)
    img2 = ensure_bgr(prev_delaunay_frame.image)
    img3 = ensure_bgr(compare_frame.image)
    
    # Place images side by side
    vis_img[:h1, :w1] = img1
    vis_img[:h2, w1:w1+w2] = img2
    vis_img[:h3, w1+w2:] = img3
    
    # Draw Delaunay edges on each frame
    def draw_graph(img, G, keypoints, offset_x=0):
        # Draw keypoints
        for kp in keypoints:
            cv2.circle(img, 
                      (int(kp[0]) + offset_x, int(kp[1])), 
                      3, (0, 255, 0), -1)
        # Draw edges
        for edge in G.edges():
            pt1 = tuple(map(int, keypoints[edge[0]]))
            pt2 = tuple(map(int, keypoints[edge[1]]))
            cv2.line(img, 
                    (pt1[0] + offset_x, pt1[1]),
                    (pt2[0] + offset_x, pt2[1]),
                    (255, 0, 0), 1)

    # Draw graphs
    draw_graph(vis_img, G_curr, curr_frame.keypoints, 0)
    draw_graph(vis_img, G_prev, prev_delaunay_frame.keypoints, w1)
    draw_graph(vis_img, G_compare, compare_frame.keypoints, w1+w2)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f'Current ({len(G_curr.edges())} edges)', 
                (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(vis_img, f'Keyframe ({len(G_prev.edges())} edges)', 
                (w1+10, 30), font, 1, (255,255,255), 2)
    cv2.putText(vis_img, f'Compare ({len(G_compare.edges())} edges)', 
                (w1+w2+10, 30), font, 1, (255,255,255), 2)
    
    # Show visualization
    cv2.imshow("Delaunay Graphs Comparison", vis_img)
    cv2.waitKey(1)
    
    return G_curr, G_prev, G_compare, curr_frame, prev_delaunay_frame, compare_frame


def get_static_dynamic_edges(curr_frame, slam):
    return 0