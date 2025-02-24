import numpy as np
import cv2
from utilities.utils_misc import delaunay_with_kps_new

def create_edge_pairs(simplicies):
    """Create edge pairs from Delaunay triangulation simplicies."""
    edge_pairs = []
    for simplex in simplicies:
        a,b,c = simplex
        for edge in [(a,b), (b,c), (c,a)]:
            if edge not in edge_pairs and (edge[1],edge[0]) not in edge_pairs:
                edge_pairs.append(edge)
    return edge_pairs

def find_matching_edges(prev_frame, curr_frame, matches):
    """Find matching edges between two frames."""
    idxs_ref, idxs_curr = matches[:, 0], matches[:, 1]
    
    # Create mappings
    idxs_ref_dict = dict(zip(idxs_ref, idxs_curr))
    
    # Get Delaunay triangulation for previous frame - using .image instead of .img
    prev_tri_simplicies, _, _ = delaunay_with_kps_new(
        prev_frame.image, prev_frame.keypoints, idxs_ref)
    
    # Create edge pairs from triangulation
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
            
    return prev_mapped_edges, curr_edges

def find_dynamic_edges(prev_frame, curr_frame, prev_edges, curr_edges, threshold=2):
    """Find edges that changed length significantly."""
    dynamic_edges = []
    
    # Calculate edge lengths
    prev_lengths = [np.linalg.norm(prev_frame.keypoints[a] - prev_frame.keypoints[b]) 
                   for a, b in prev_edges]
    curr_lengths = [np.linalg.norm(curr_frame.keypoints[a] - curr_frame.keypoints[b]) 
                   for a, b in curr_edges]
    
    # Find edges with significant length changes
    for i, (prev_len, curr_len) in enumerate(zip(prev_lengths, curr_lengths)):
        print(prev_len, curr_len)
        if abs(prev_len - curr_len) > threshold:

            dynamic_edges.append(i)
            
    return dynamic_edges

def find_connected_components(edges, dynamic_edges):
    """Find connected components in graph excluding dynamic edges."""
    # Create adjacency matrix excluding dynamic edges
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

def create_dynamic_mask(frame, components):
    """Create binary mask marking dynamic regions."""
    h, w = frame.image.shape[:2]
    # Create uint8 mask instead of boolean
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Skip largest component (assumed static)
    for component in components[1:]:
        points = np.array([frame.keypoints[i] for i in component])
        hull = cv2.convexHull(points.astype(np.int32))
        # Fill with 0 for dynamic regions
        cv2.fillPoly(mask, [hull], 0)
    
    # Convert back to boolean mask (True for static, False for dynamic)
    return mask > 0

def visualize_edges(prev_frame, curr_frame, prev_edges, curr_edges, dynamic_edges, components=None):
    """Visualize edge analysis results."""
    # Create side-by-side visualization - using .image instead of .img
    vis_img = np.hstack([prev_frame.image, curr_frame.image])
    w = prev_frame.image.shape[1]
    
    # Draw edges
    for i in range(len(prev_edges)):
        # Color: red for dynamic, green for static
        color = (0, 0, 255) if i in dynamic_edges else (0, 255, 0)
        
        # Draw in previous frame
        a, b = prev_edges[i]
        pt1 = tuple(map(int, prev_frame.keypoints[a]))
        pt2 = tuple(map(int, prev_frame.keypoints[b]))
        cv2.line(vis_img, pt1, pt2, color, 2)
        
        # Draw in current frame
        a, b = curr_edges[i]
        pt1 = (int(curr_frame.keypoints[a][0] + w), int(curr_frame.keypoints[a][1]))
        pt2 = (int(curr_frame.keypoints[b][0] + w), int(curr_frame.keypoints[b][1]))
        cv2.line(vis_img, pt1, pt2, color, 2)
    
    # Draw component hulls
    if components:
        colors = [(255,0,0), (0,255,255), (255,0,255)]  # Different colors for components
        for idx, comp in enumerate(components[:3]):  # Show top 3 components
            points = np.array([curr_frame.keypoints[i] for i in comp])
            hull = cv2.convexHull(points.astype(np.int32))
            
            # Offset hull for right image
            hull_offset = hull.copy()
            hull_offset[:,:,0] += w
            
            cv2.polylines(vis_img, [hull_offset], True, colors[idx], 2)
            
    cv2.imshow("Edge Analysis", vis_img)
    cv2.waitKey(1)
    
    return vis_img

def visualize_dynamic_components(prev_frame, curr_frame, components):
    """Visualize only the dynamic components (all except the largest component)."""
    # Create side-by-side visualization
    vis_img = np.hstack([prev_frame.image, curr_frame.image])
    w = prev_frame.image.shape[1]
    
    if components and len(components) > 1:
        # Skip the largest component (components[0]) and visualize the rest
        dynamic_components = components[1:]
        
        # Colors for different dynamic components
        colors = [(0,0,255), (255,0,255), (0,255,255), (255,165,0), (255,0,0)]
        
        for idx, comp in enumerate(dynamic_components):
            # Get color (cycle through colors if more components than colors)
            color = colors[idx % len(colors)]
            
            # Get points for this component
            points = np.array([curr_frame.keypoints[i] for i in comp])
            
            # Draw convex hull
            hull = cv2.convexHull(points.astype(np.int32))
            
            # Draw filled hull with some transparency
            overlay = vis_img.copy()
            hull_offset = hull.copy()
            hull_offset[:,:,0] += w  # Offset for right image
            cv2.fillPoly(overlay, [hull_offset], color)
            cv2.addWeighted(overlay, 0.3, vis_img, 0.7, 0, vis_img)
            
            # Draw hull boundary
            cv2.polylines(vis_img, [hull_offset], True, color, 2)
            
            # Add component size text
            centroid = np.mean(hull_offset, axis=0)
            cv2.putText(vis_img, f"Size: {len(comp)}", 
                       (int(centroid[0][0]), int(centroid[0][1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Dynamic Components", vis_img)
    cv2.waitKey(1)
    
    return vis_img
