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

def matches_with_k_frames_away_with_prev_delaunay_edges(frame, slam, k):
    """Visualize three frames showing only common keypoints matched across all three in green.
    
    Args:
        frame: Current frame
        slam: SLAM system containing previous frames and matcher
        k: Number of frames to look back
    
    Returns:
        vis_img: Visualization image showing three frames side by side with only common keypoints
    """
    # Get the frames
    frame = frame
    try:
        frame_with_last_delaunay = slam.map.get_last_keyframe()
    except IndexError:
        print(slam.map.keyframes)
        print("No keyframes in map")
        return None
    frame_k_frames_away = get_frame_from_pyslam_dataloader(slam.dataset, slam.groundtruth, frame.id - k, slam.config)

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
    common_curr_indices = set()
    for curr_idx, prev_idx in curr_prev_dict.items():
        if curr_idx in curr_k_dict:  # Matches with k-frame
            k_idx = curr_k_dict[curr_idx]
            if prev_idx in prev_k_dict and prev_k_dict[prev_idx] == k_idx:  # Consistent triangle
                common_curr_indices.add(curr_idx)

    # Draw only common keypoints
    # Current frame (left)
    for i, kp in enumerate(frame.keypoints):
        if i in common_curr_indices:
            draw_keypoint(vis_img, kp, GREEN)

    # Previous keyframe (middle)
    for i, kp in enumerate(frame_with_last_delaunay.keypoints):
        if i in [curr_prev_dict[curr_idx] for curr_idx in common_curr_indices]:
            draw_keypoint(vis_img, kp, GREEN, w1)

    # K-frames-away frame (right)
    for i, kp in enumerate(frame_k_frames_away.keypoints):
        if i in [curr_k_dict[curr_idx] for curr_idx in common_curr_indices]:
            draw_keypoint(vis_img, kp, GREEN, w1+w2)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis_img, f'Current Frame (id:{frame.id})', (10, 30), font, 1, (255,255,255), 2)
    cv2.putText(vis_img, f'Last Keyframe (id:{frame_with_last_delaunay.id})', (w1+10, 30), font, 1, (255,255,255), 2)
    cv2.putText(vis_img, f'K-Frame Away (id:{frame_k_frames_away.id})', (w1+w2+10, 30), font, 1, (255,255,255), 2)

    # Show visualization
    cv2.imshow("Common Keypoints Visualization", vis_img)
    cv2.waitKey(1)

    return vis_img