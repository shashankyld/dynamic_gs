from core.map import Map
from utilities.utils_draw import visualize_global_map, visualize_local_map
import gtsam
import numpy as np
from gtsam.symbol_shorthand import X, L  # X: poses, L: landmarks

def create_noise_models():
    """Create noise models for different factors."""
    # Prior noise
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))  # rotation and position
    
    # Odometry noise
    odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))  # rotation and position
    
    # Landmark measurement noise
    point_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # pixel noise
    
    return prior_noise, odom_noise, point_noise

def create_calibration(keyframe):
    """Create GTSAM calibration from keyframe intrinsics."""
    return gtsam.Cal3_S2(
        keyframe.fx, keyframe.fy,
        0.0,  # skew
        keyframe.cx, keyframe.cy
    )

def pose_to_gtsam(pose_mat):
    """Convert 4x4 pose matrix to GTSAM Pose3."""
    R = pose_mat[:3, :3]
    t = pose_mat[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), t)

def optimize_graph(graph, initial):
    """Optimize the factor graph."""
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    return result

def select_well_constrained_points(loaded_map, local_kf_ids, min_observations=2, max_points=1000):
    """Select points that are well-observed by local keyframes."""
    point_observations = {}
    
    # Debug information
    print("\nDebug information for point selection:")
    for kf_id in local_kf_ids:
        kf = loaded_map.keyframes[kf_id]
        print(f"KF {kf_id}:")
        print(f"- Has visible_map_points: {hasattr(kf, 'visible_map_points')}")
        if hasattr(kf, 'visible_map_points'):
            print(f"- Number of visible points: {len(kf.visible_map_points)}")
            print(f"- Has keypoints: {kf.keypoints is not None}")
            if kf.keypoints is not None:
                print(f"- Number of keypoints: {len(kf.keypoints)}")
    
    # Get points visible in any local keyframe
    local_points = set()
    for kf_id in local_kf_ids:
        kf = loaded_map.keyframes[kf_id]
        if hasattr(kf, 'visible_map_points') and kf.visible_map_points:
            local_points.update(kf.visible_map_points)
    
    print(f"\nFound {len(local_points)} total visible points across local keyframes")
    
    # Count observations for each point
    for point_id in local_points:
        if point_id in loaded_map.map_points:
            point = loaded_map.map_points[point_id]
            # Count how many local keyframes observe this point
            local_observations = sum(1 for kf_id in local_kf_ids 
                                  if kf_id in point.observing_keyframes)
            if local_observations >= min_observations:
                point_observations[point_id] = local_observations
    
    print(f"Found {len(point_observations)} points with {min_observations}+ observations")
    
    # Sort points by number of observations
    sorted_points = sorted(point_observations.items(), key=lambda x: x[1], reverse=True)
    
    # Take top N points
    selected_points = set(point_id for point_id, _ in sorted_points[:max_points])
    
    # More detailed output
    if sorted_points:
        print("\nTop 5 points observation counts:")
        for point_id, count in sorted_points[:5]:
            point = loaded_map.map_points[point_id]
            print(f"Point {point_id}: {count} observations, seen in KFs: {sorted(point.observing_keyframes)}")
    
    return selected_points

def create_and_optimize_ba_graph(loaded_map):
    """Create and optimize bundle adjustment graph for local keyframes only."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    
    prior_noise, odom_noise, point_noise = create_noise_models()
    
    # Get local keyframe IDs and sort them
    local_kf_ids = sorted(loaded_map.local_keyframes)
    if not local_kf_ids:
        print("No local keyframes to optimize!")
        return None, None, None
    
    print(f"Processing {len(local_kf_ids)} local keyframes")
    
    # Select well-constrained points for optimization
    selected_points = select_well_constrained_points(loaded_map, local_kf_ids)
    if not selected_points:
        print("No well-constrained points found!")
        # Print some debug info about the map state
        print("\nMap state:")
        print(f"Local keyframes: {local_kf_ids}")
        if local_kf_ids:
            example_kf = loaded_map.keyframes[local_kf_ids[0]]
            print(f"Example keyframe {local_kf_ids[0]} attributes:")
            print(f"- Has pose: {example_kf.pose is not None}")
            print(f"- Has keypoints: {example_kf.keypoints is not None}")
            print(f"- Has visible_map_points: {hasattr(example_kf, 'visible_map_points')}")
        return None, None, None
    
    # Create mapping from global keyframe ID to local optimization index
    kf_id_to_index = {kf_id: idx for idx, kf_id in enumerate(local_kf_ids)}
    
    # Add prior factor for the first local keyframe
    first_kf = loaded_map.keyframes[local_kf_ids[0]]
    first_pose = pose_to_gtsam(first_kf.pose)
    graph.add(gtsam.PriorFactorPose3(X(0), first_pose, prior_noise))
    initial.insert(X(0), first_pose)
    
    # Create calibration from first keyframe
    K = create_calibration(first_kf)
    
    # Add odometry factors between consecutive local keyframes
    for i in range(1, len(local_kf_ids)):
        prev_kf = loaded_map.keyframes[local_kf_ids[i-1]]
        curr_kf = loaded_map.keyframes[local_kf_ids[i]]
        
        curr_pose = pose_to_gtsam(curr_kf.pose)
        initial.insert(X(i), curr_pose)
        
        prev_to_curr = pose_to_gtsam(prev_kf.pose).between(curr_pose)
        graph.add(gtsam.BetweenFactorPose3(
            X(i-1), X(i), prev_to_curr, odom_noise))
    
    # Add only selected local map points as landmarks
    landmark_count = 0
    point_id_to_landmark = {}  # Mapping from point ID to landmark index
    
    for point_id in selected_points:
        map_point = loaded_map.map_points[point_id]
        initial.insert(L(landmark_count), map_point.position)
        point_id_to_landmark[point_id] = landmark_count
        
        # Add projection factors only for local keyframes that observe this point
        observations = 0
        for kf_id in map_point.observing_keyframes:
            if kf_id in local_kf_ids:  # Only if keyframe is local
                kf = loaded_map.keyframes[kf_id]
                local_idx = kf_id_to_index[kf_id]
                
                if kf.keypoints is not None and kf.visible_map_points is not None:
                    if point_id in kf.visible_map_points:
                        try:
                            kp_idx = list(kf.visible_map_points).index(point_id)
                            if kp_idx < len(kf.keypoints):
                                measurement = gtsam.Point2(kf.keypoints[kp_idx])
                                factor = gtsam.GenericProjectionFactorCal3_S2(
                                    measurement,
                                    point_noise,
                                    X(local_idx),
                                    L(landmark_count),
                                    K)
                                graph.add(factor)
                                observations += 1
                        except ValueError:
                            continue
                            
        if observations > 0:  # Only increment if point was actually used
            landmark_count += 1
    
    print(f"Optimizing local window with {len(local_kf_ids)} keyframes")
    print(f"Using {landmark_count} landmarks")
    print(f"Graph has {graph.size()} factors and {initial.size()} variables")
    
    # Optimize
    result = optimize_graph(graph, initial)
    
    return result, local_kf_ids, selected_points

def update_map_with_optimization(loaded_map, result, local_kf_ids):
    """Update map with optimized poses and points (local only)."""
    if result is None or local_kf_ids is None:
        return
        
    # Update local keyframe poses
    for i, kf_id in enumerate(local_kf_ids):
        if result.exists(X(i)):
            optimized_pose = result.atPose3(X(i))
            loaded_map.keyframes[kf_id].pose = optimized_pose.matrix()
    
    # Update local map points
    landmark_count = 0
    for point_id in loaded_map.local_map_points:
        if result.exists(L(landmark_count)):
            optimized_point = result.atPoint3(L(landmark_count))
            loaded_map.map_points[point_id].position = optimized_point
            landmark_count += 1

if __name__ == '__main__':
    map_path = "/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/logs/slam_map/20250224_150050/map.pkl"
    try:
        print("Loading map from:", map_path)
        loaded_map = Map.load(map_path)
        
        # Print initial map statistics
        print("\nInitial Map Statistics:")
        print(f"Number of keyframes: {len(loaded_map.keyframes)}")
        print(f"Number of map points: {len(loaded_map.map_points)}")
        print(f"Number of local map points: {len(loaded_map.local_map_points)}")

        # Visualize the loaded map
        visualize_local_map(loaded_map,
                            title="Loaded Local Map Visualization",
                            dense=True)
        
        # Create and optimize local BA graph
        result, local_kf_ids, selected_points = create_and_optimize_ba_graph(loaded_map)
        
        if result is not None:
            # Update map with optimized values
            update_map_with_optimization(loaded_map, result, local_kf_ids)
            print("\nLocal Bundle Adjustment completed!")
            print(f"Optimized {len(local_kf_ids)} local keyframes")
            print(f"Used {len(selected_points)} map points")
            
            # Visualize the optimized map
            visualize_local_map(loaded_map,
                               title=f"Optimized Local Map - {len(local_kf_ids)} KFs, {len(selected_points)} points",
                               dense=True)
        else:
            print("No optimization performed!")
    
    except Exception as e:
        print(f"Error in bundle adjustment: {e}")
        raise e
