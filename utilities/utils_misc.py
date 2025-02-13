
import numpy as np
import cv2
from scipy.spatial import Delaunay
from utilities.utils_depth import depth2pointcloud
import open3d as o3d


def estimate_pose_icp(ref_pc, cur_pc, threshold = 0.1, init_T = np.eye(4)):
    """
    Estimates the camera pose using ICP.

    Args:
        ref_pc: Reference point cloud (Nx3 numpy array).
        cur_pc: Current point cloud (Mx3 numpy array).
        threshold: Distance threshold for ICP.
        init_T: Initial transformation (4x4 numpy array).

    Returns:
        success: Boolean indicating success.
        T: Transformation matrix (4x4).
    """

    # Create Open3D point cloud objects
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(ref_pc)

    cur_pcd = o3d.geometry.PointCloud()
    cur_pcd.points = o3d.utility.Vector3dVector(cur_pc)

    # Create Open3D ICP object
    icp = o3d.pipelines.registration.registration_icp(
        ref_pcd, cur_pcd, threshold, init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    # Get the transformation matrix
    T = icp.transformation

    return T

def estimate_pose_ransac(ref_depth, ref_img, kpts0, kpts1, matches, camera_matrix, dist_coeffs=None):
    """
    Estimates the camera pose using RANSAC and PnP.

    Args:
        ref_depth: Depth image of the reference frame (HxW numpy array).
        ref_img:  RGB image of ref frame.
        kpts0: Keypoints in the reference frame (Nx2 numpy array).
        kpts1: Keypoints in the current frame (Mx2 numpy array).
        matches: Matches (Kx2 numpy array of indices).
        camera_matrix: Camera intrinsics (3x3 numpy array).
        dist_coeffs: Distortion coefficients (optional).

    Returns:
        success: Boolean indicating success.
        R: Rotation matrix (3x3).
        t: Translation vector (3x1).
        inliers: Indices of inlier matches.
    """

    print("Matches", matches)

    # 1.  3D point generation (from reference frame)
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    points3d = depth2pointcloud(ref_depth, ref_img, fx, fy, cx, cy, max_depth=100000.0, min_depth=0.0).points  # (HxWx
    print("points3d", points3d)
    # # Visually check the points3d
    # o3d_points = o3d.geometry.PointCloud()
    # o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    # o3d_points.points = o3d.utility.Vector3dVector(points3d)

    # o3d.visualization.draw_geometries([o3d_points, o3d_axis])


    print("points3d", points3d.shape)
    print("Type of points3d", type(points3d))
    # 2. Create 3D-2D correspondences
    points3d_matched = points3d[matches[:, 0]]  # 3D points (reference frame)
    points2d_matched = kpts1[matches[:, 1]]  # 2D points (current frame)

    # Remove invalid depth points
    # Check for valid z (depth) >0 and <100
    valid_mask = points3d_matched[:, 2] > 0.0 # Depth > 0
    valid_mask = np.logical_and(valid_mask, points3d_matched[:, 2] < 100.0)  # Depth < 100
    points3d_matched = points3d_matched[valid_mask]
    # Visualize the points3d_matched
    o3d_points_matched = o3d.geometry.PointCloud()
    o3d_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0])
    o3d_points_matched.points = o3d.utility.Vector3dVector(points3d_matched)
    o3d.visualization.draw_geometries([o3d_points_matched, o3d_axis])

    points2d_matched = points2d_matched[valid_mask]

    # 3. RANSAC + PnP
    if len(points3d_matched) >= 4:  # Need at least 4 points for PnP
        # success, rvec, tvec, inliers = cv2.solvePnPRansac(
        #     points3d_matched,
        #     points2d_matched,
        #     camera_matrix,
        #     dist_coeffs,
        #     iterationsCount=100, # RANSAC iterations
        #     reprojectionError=13.0,  # Reprojection error threshold
        #     confidence=0.99, # Confidence level
        #     flags=cv2.SOLVEPNP_ITERATIVE  # Solver method
        # )

        success, rvec, tvec, inliers = cv2.solvePnPRansac(points3d_matched, points2d_matched, camera_matrix, dist_coeffs,iterationsCount=100, reprojectionError=8.0, confidence=0.99, flags=cv2.SOLVEPNP_EPNP)  # Use EPnP instead of iterative PnP)

        if success:
            R, _ = cv2.Rodrigues(rvec)  # Rotation vector to matrix
            t = tvec.reshape(3, 1)
            return success, R, t, inliers
        else:
            return False, None, None, None
    else:
       return False, None, None, None

def remove_duplicates_from_index_arrays(idxs_ref, idxs_cur):
    """
    Removes duplicates from idxs_ref and corresponding elements from idxs_cur,
    then removes any remaining duplicates from idxs_cur and corresponding elements from idxs_ref.
    Handles empty idxs_cur.

    Args:
        idxs_ref: Indices of reference frame keypoints.
        idxs_cur: Indices of current frame keypoints.

    Returns:
        Updated idxs_ref, idxs_cur.
    """

    if len(idxs_cur) == 0:  # Handle empty idxs_cur
        print("Warning: idxs_cur is empty. Returning original arrays.")
        return idxs_ref, idxs_cur

    # 1. Identify and Remove Duplicates in idxs_ref
    unique_idxs_ref, unique_indices = np.unique(idxs_ref, return_index=True)
    duplicate_indices = np.setdiff1d(np.arange(len(idxs_ref)), unique_indices)

    mask = np.ones(len(idxs_ref), dtype=bool)
    mask[duplicate_indices] = False

    idxs_ref_updated = idxs_ref[mask]
    idxs_cur_updated = idxs_cur[mask]

    # 2. Identify and Remove Remaining Duplicates in idxs_cur
    unique_idxs_cur, unique_indices_cur = np.unique(idxs_cur_updated, return_index=True)
    duplicate_indices_cur = np.setdiff1d(np.arange(len(idxs_cur_updated)), unique_indices_cur)

    mask_cur = np.ones(len(idxs_cur_updated), dtype=bool)
    mask_cur[duplicate_indices_cur] = False

    idxs_cur_updated = idxs_cur_updated[mask_cur]
    idxs_ref_updated = idxs_ref_updated[mask_cur]

    return idxs_ref_updated, idxs_cur_updated


def delaunay_with_kps_new(img, kps, idxs, all_kps=False):
    if len(idxs) == 0:
        print("Warning: No keypoints available for Delaunay triangulation.")
        return None

    if not all_kps:
        kps = kps[idxs]

    if len(kps) < 3:  # Delaunay needs at least 3 points
        print("Warning: Not enough points for triangulation.")
        return None

    # Convert to int if necessary
    kps = kps.astype(np.int32)

    # Perform Delaunay triangulation
    tri = Delaunay(kps)
    tri_indices = tri.simplices
    tri_vertices = kps[tri_indices]

    # Draw triangles
    for tri_vert in tri_vertices:
        tri_vert = np.array(tri_vert, dtype=np.int32)
        cv2.polylines(img, [tri_vert], isClosed=True, color=(0, 255, 0), thickness=1)

    print("Delaunay triangulation completed.")
    return tri_indices, tri_vertices, img

def delaunay_with_kps(frame, idxs, all_kps=False):
    kps = frame.kps
    if len(idxs) == 0:
        print("Warning: No keypoints available for Delaunay triangulation.")
        return None

    if not all_kps:
        kps = kps[idxs]
    

    if len(kps) < 3:  # Delaunay needs at least 3 points
        print("Warning: Not enough points for triangulation.")
        return None
    
    
    # Convert to int if necessary
    kps = kps.astype(np.int32)

    # Perform Delaunay triangulation
    tri = Delaunay(kps) # <scipy.spatial._qhull.Delaunay object at 0x7eff1746d040>
    tri_indices = tri.simplices # [[215 590 414], [213 212 414]...] # theses are the indices of the points in kps that form the triangles
    tri_vertices = kps[tri_indices] 
    '''
    # These are the vertices of the triangles in the image in pixel coordinates
    tri_vertices =
    [
        [
            [ 57 443]
            [ 70 450]
            [ 70 451]
        ]

        [
            [ 71 451]
            [ 86 456]
            [ 70 451]
        ]
    ]
    '''

    # Draw triangles
    img = frame.img.copy()
    for tri_vert in tri_vertices:
        tri_vert = np.array(tri_vert, dtype=np.int32)  # Ensure integer format
        cv2.polylines(img, [tri_vert], isClosed=True, color=(0, 255, 0), thickness=1)

    print("Delaunay triangulation completed.")
    print("tri", tri)
    print("tri_indices", tri_indices)
    print("tri_vertices", tri_vertices)
    return tri_indices, tri_vertices, img


    

def convert_frame_to_delaunay_dict(frame, idxs):
    kpsu = frame.kpsu[idxs] # [[ 97.02989  306.17648 ], [ 45.978912 326.03632 ], ...]
    kps  = frame.kps[idxs]  # [[ 57. 321.],  [ 71. 299.], ...]
    kpsn = frame.kpsn[idxs] # [[-0.46763112  0.15167243],  [-0.44217141  0.1117238 ], ...]
    kps_depth = frame.depths[idxs] # [1.947 1.926 1.756 ...]
    delaunay_with_kps_ = delaunay_with_kps(frame, idxs)
    return {
        'frame': frame,
        'idxs': idxs,
        'kpsu': kpsu,
        'kps': kps,
        'kpsn': kpsn,
        'kps_depth': kps_depth,
        'delaunay_with_kps': delaunay_with_kps_
    }
    

def delaunay_visualization(prev_delaunay_img, curr_delaunay_img):
    # Horizontally stack the images
    img = np.hstack((prev_delaunay_img, curr_delaunay_img))
    cv2.imshow("Delaunay Triangulation", img)
    cv2.waitKey(2)  


# def get_common_edges(prev_frame_dict, cur_frame_dict):
#     # Check if Delaunay triangulation data exists for both frames
#     prev_delaunay_data = prev_frame_dict.get('delaunay_with_kps')
#     curr_delaunay_data = cur_frame_dict.get('delaunay_with_kps')

#     prev_idxs = prev_frame_dict.get('idxs')
#     curr_idxs = cur_frame_dict.get('idxs')

#     print("Prev idxs:", len(prev_idxs))
#     print("Curr idxs:", len(curr_idxs))

#     # Create a dict with prev idxs as keys and curr idxs as values
#     idxs_dict = dict(zip(prev_idxs, curr_idxs))
#     # print("Idxs dict:", idxs_dict.keys())
#     print("len Idxs dict:", len(idxs_dict))



#     if prev_delaunay_data is None or curr_delaunay_data is None:
#         return []
    
#     # Extract triangle indices from both frames
#     prev_tri_indices = prev_delaunay_data[0]
#     curr_tri_indices = curr_delaunay_data[0]

#     # print("Prev tri indices:", prev_tri_indices)
#     # print("Curr tri indices:", curr_tri_indices)
    
#     # Helper function to extract edges from triangle indices
#     def extract_edges(tri_indices):
#         edges = set()
#         for simplex in tri_indices:
#             # Generate all three edges of the triangle and store as sorted tuples
#             a, b, c = simplex
#             edges.add(tuple(sorted((a, b))))
#             edges.add(tuple(sorted((b, c))))
#             edges.add(tuple(sorted((c, a))))
#         return edges
    
    
#     # Get edges for previous and current frames
#     prev_edges = extract_edges(prev_tri_indices)
#     curr_edges = extract_edges(curr_tri_indices)

#     print("Length of prev edges:", len(prev_edges))
#     print("Length of curr edges:", len(curr_edges))
    
#     # print("Prev edges:", prev_edges)
#     # print("######################################")
#     # print("######################################")
#     # print("######################################")
#     # print("######################################")
#     # print("Curr edges:", curr_edges)
    
#     # For each edge in previous frame, create a probable edge in the current frame using idxs_dict
#     common_edges = set()
#     vertices_prev = set()
#     for edge in prev_edges: 
#         a, b = edge
#         vertices_prev.add(a)
#         vertices_prev.add(b)

#     print("Vertices prev:", len(vertices_prev))
#     # for every vertex in the vertices_prev, check if it is in the idxs_dict 
#     count = 0
#     for vertex in vertices_prev:
#         if vertex in idxs_dict:
#             True
#             count += 1
#     print("Count:", count)

#     # # Now intersect the two sets of fake and real edges to get the common edges
#     # common_edges = common_edges.intersection(curr_edges)
#     print("Length of common edges:", len(common_edges))


            

    # vertices_prev = set()
    # for edge in prev_edges:
    #     a, b = edge
    #     if a in idxs_dict and b in idxs_dict:
    #         vertices_prev.add(a)
    #         vertices_prev.add(b)

    #     fake_edge_a = idxs_dict.get(a)
    #     fake_edge_b = idxs_dict.get(b)
        
    # return list(common_edges)


def get_common_edges(prev_frame_dict, cur_frame_dict):
    prev_delaunay_data = prev_frame_dict.get('delaunay_with_kps')
    curr_delaunay_data = cur_frame_dict.get('delaunay_with_kps')

    prev_idxs = prev_frame_dict.get('idxs')
    curr_idxs = cur_frame_dict.get('idxs')

    if prev_delaunay_data is None or curr_delaunay_data is None:
        return []

    idxs_dict = dict(zip(prev_idxs, curr_idxs))
    reverse_curr_dict = {orig_idx: pos for pos, orig_idx in enumerate(curr_idxs)}

    prev_tri_indices = prev_delaunay_data[0]
    curr_tri_indices = curr_delaunay_data[0]

    def extract_edges(tri_indices):
        edges = set()
        for simplex in tri_indices:
            a, b, c = simplex
            edges.add(tuple(sorted((a, b))))
            edges.add(tuple(sorted((b, c))))
            edges.add(tuple(sorted((c, a))))
        return edges

    prev_edges = extract_edges(prev_tri_indices)
    curr_edges = extract_edges(curr_tri_indices)

    common_edges = set()

    for edge in prev_edges:
        a_prev, b_prev = edge
        orig_a_prev = prev_idxs[a_prev]
        orig_b_prev = prev_idxs[b_prev]

        orig_a_curr = idxs_dict.get(orig_a_prev)
        orig_b_curr = idxs_dict.get(orig_b_prev)

        if orig_a_curr is None or orig_b_curr is None:
            continue

        pos_a_curr = reverse_curr_dict.get(orig_a_curr)
        pos_b_curr = reverse_curr_dict.get(orig_b_curr)

        if pos_a_curr is None or pos_b_curr is None:
            continue

        sorted_edge = tuple(sorted((pos_a_curr, pos_b_curr)))
        if sorted_edge in curr_edges:
            common_edges.add((orig_a_prev, orig_b_prev, orig_a_curr, orig_b_curr))
    
    print("Length of prev idxs:", len(prev_idxs))
    print("Length of curr idxs:", len(curr_idxs))
    print("Length of prev edges:", len(prev_edges))
    print("Length of common edges:", len(common_edges))
    print("Common edges:", common_edges)


    # For each common edge, corresponding length in prev and curr frame
    prev_lengths = []
    curr_lengths = []
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        prev_lengths.append(np.linalg.norm(prev_frame_dict['frame'].depths[a_prev] - prev_frame_dict['frame'].depths[b_prev]))
        curr_lengths.append(np.linalg.norm(cur_frame_dict['frame'].depths[a_curr] - cur_frame_dict['frame'].depths[b_curr]))


    return list(common_edges), prev_lengths, curr_lengths

def draw_common_edges(prev_frame, curr_frame,common_edges):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (0, 255, 0), 1) 
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (0, 255, 0), 1)
    cv2.imshow("Common Edges", img)
    cv2.waitKey(2)


def draw_common_edges_with_lengths(prev_frame, curr_frame, common_edges, prev_lengths, curr_lengths):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    # Draw all common edges, but for few of them draw the length as well in red 
    for i, edge in enumerate(common_edges):
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (0, 255, 0), 1)
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (0, 255, 0), 1)
        if i % 100 == 0:
            cv2.putText(img, f"{prev_lengths[i]:.2f}", tuple((a_prev + b_prev) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # Size of text is 0.5, color is red, thickness is 1
            cv2.putText(img, f"{curr_lengths[i]:.2f}", tuple((a_curr + b_curr) // 2 + [prev_frame.img.shape[1], 0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow("Common Edges with Lengths", img)
    cv2.waitKey(2)




def get_dynamic_edges(curr_frame, prev_frame, common_edges, threshold=0.1):
    """ 
    Returns the dynamic edges between the current and previous frame.
    All the edge lengths are calculated in the current frame and the previous frame. in the world frame/sensor frame- we can directly check in sensor frame as we have rgbd data
    for all the edges in the common_edges, the ones that change their length are considered as dynamic edges. by a margin of 0.1
    """
    dynamic_edges = []
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev_depth = prev_frame.depths[a_prev]
        b_prev_depth = prev_frame.depths[b_prev]
        a_curr_depth = curr_frame.depths[a_curr]
        b_curr_depth = curr_frame.depths[b_curr]

        prev_edge_length = np.linalg.norm(a_prev_depth - b_prev_depth)
        curr_edge_length = np.linalg.norm(a_curr_depth - b_curr_depth)

        if abs(curr_edge_length - prev_edge_length) > threshold:
            dynamic_edges.append(edge)

    # For each dynamic edge, corresponding length in prev and curr frame
    prev_lengths = []
    curr_lengths = []
    for edge in dynamic_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        prev_lengths.append(np.linalg.norm(prev_frame.depths[a_prev] - prev_frame.depths[b_prev]))
        curr_lengths.append(np.linalg.norm(curr_frame.depths[a_curr] - curr_frame.depths[b_curr]))


    return dynamic_edges, prev_lengths, curr_lengths

def draw_dynamic_edges(prev_frame, curr_frame, dynamic_edges):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    for edge in dynamic_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (0, 0, 255), 1)
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (0, 0, 255), 1)
    cv2.imshow("Dynamic Edges", img)
    cv2.waitKey(2)

def draw_dynamic_edges_with_lengths(prev_frame, curr_frame, dynamic_edges, prev_lengths, curr_lengths):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    # Draw all dynamic edges, but for few of them draw the length as well in red 
    for i, edge in enumerate(dynamic_edges):
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (0, 0, 255), 1)
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (0, 0, 255), 1)
        if i % 100 == 0:
            cv2.putText(img, f"{prev_lengths[i]:.2f}", tuple((a_prev + b_prev) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1) # Size of text is 0.5, color is red, thickness is 1
            cv2.putText(img, f"{curr_lengths[i]:.2f}", tuple((a_curr + b_curr) // 2 + [prev_frame.img.shape[1], 0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow("Dynamic Edges with Lengths", img)
    cv2.waitKey(2)


import networkx as nx


def get_static_edges(common_edges, dynamic_edges):
    """
    Returns the static edges between the current and previous frame.
    """
    common_edges_set = set(common_edges)
    dynamic_edges_set = set(dynamic_edges)
    static_edges = common_edges_set - dynamic_edges_set
    return list(static_edges)

def draw_static_edges(prev_frame, curr_frame, static_edges):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    for edge in static_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        a_prev = prev_frame.kps[a_prev].astype(np.int32)
        b_prev = prev_frame.kps[b_prev].astype(np.int32)
        a_curr = curr_frame.kps[a_curr].astype(np.int32)
        b_curr = curr_frame.kps[b_curr].astype(np.int32)
        cv2.line(img, tuple(a_prev), tuple(b_prev), (255, 0, 0), 1)
        cv2.line(img, tuple(a_curr + [prev_frame.img.shape[1], 0]), tuple(b_curr + [prev_frame.img.shape[1], 0]), (255, 0, 0), 1)
    cv2.imshow("Static Edges", img)
    cv2.waitKey(2)



## UTILS FOR CONVERTING EDGES TO GRAPH and finding connected components after removing dynamic edges from the graph



import networkx as nx

def get_connected_components_from_edges(common_edges, prev_frame, curr_frame):
    # Create an empty graph
    G = nx.Graph()
    
    # Add nodes to the graph from keypoints (you could use indices of keypoints as nodes)
    for idx in range(len(prev_frame.kps)):
        G.add_node(idx)  # Adding nodes for previous frame keypoints
    for idx in range(len(curr_frame.kps)):
        G.add_node(len(prev_frame.kps) + idx)  # Adding nodes for current frame keypoints
    
    # Add edges between corresponding keypoints
    for edge in common_edges:
        a_prev, b_prev, a_curr, b_curr = edge
        # Add edges between matching keypoints: one for previous and one for current
        G.add_edge(a_prev, a_curr + len(prev_frame.kps))  # From prev to curr
        G.add_edge(b_prev, b_curr + len(prev_frame.kps))  # From prev to curr
    
    # Find the connected components
    connected_components = list(nx.connected_components(G))
    
    return connected_components

# Example usage
common_edges = [(0, 1, 2, 3), (4, 5, 6, 7)]  # Example common edges
# Assuming prev_frame and curr_frame are available
# connected_components = get_connected_components_from_edges(common_edges, prev_frame, curr_frame)

# Print the connected components
# print("Connected Components:", connected_components)

def draw_connected_components(prev_frame, curr_frame, connected_components):
    # Horizontally stack the images
    img = np.hstack((prev_frame.img, curr_frame.img))
    
    # Draw the connected components
    for component in connected_components:
        for idx in component:
            if idx < len(prev_frame.kps):
                cv2.circle(img, tuple(prev_frame.kps[idx].astype(np.int32)), 3, (0, 255, 0), -1)
            else:
                cv2.circle(img, tuple(curr_frame.kps[idx - len(prev_frame.kps)].astype(np.int32) + [prev_frame.img.shape[1], 0]), 3, (0, 255, 0), -1)
    
    cv2.imshow("Connected Components", img)
    cv2.waitKey(2)


