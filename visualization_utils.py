import open3d as o3d
import numpy as np

def create_visualization():
    """Create and initialize visualizer"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    return vis

def visualize_point_cloud_map(point_clouds, poses, voxel_size=0.05):
    """Create and visualize global point cloud map"""
    vis = create_visualization()
    
    # Create global map
    global_map = create_global_map(point_clouds, poses, voxel_size)
    if global_map is not None:
        vis.add_geometry(global_map)
    
    # Add trajectory and poses
    add_trajectory_visualization(vis, poses)
    
    # Add world frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(world_frame)
    
    # Set view
    set_default_view(vis)
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def create_global_map(point_clouds, poses, voxel_size=0.05):
    """Create global point cloud map"""
    all_points = []
    all_colors = []
    
    for frame_id, points in point_clouds.items():
        if frame_id not in poses:
            continue
            
        pts = points.points
        colors = points.colors
        
        # Transform points
        pts_homog = np.hstack((pts, np.ones((pts.shape[0], 1))))
        transformed_pts = (poses[frame_id] @ pts_homog.T).T[:, :3]
        
        all_points.append(transformed_pts)
        all_colors.append(colors)
        
    if not all_points:
        return None
        
    # Create and downsample global map
    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    global_map.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    
    if voxel_size > 0:
        global_map = global_map.voxel_down_sample(voxel_size=voxel_size)
        
    return global_map

def add_trajectory_visualization(vis, poses):
    """Add camera poses and trajectory visualization"""
    # Add camera frames
    for pose in poses.values():
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        cam_frame.transform(pose)
        vis.add_geometry(cam_frame)
    
    # Add trajectory
    points = [pose[:3, 3] for pose in poses.values()]
    if points:
        trajectory = o3d.geometry.LineSet()
        trajectory.points = o3d.utility.Vector3dVector(points)
        trajectory.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(points)-1)])
        trajectory.paint_uniform_color([1, 0, 0])
        vis.add_geometry(trajectory)

def set_default_view(vis):
    """Set default view parameters"""
    vis.get_view_control().set_zoom(0.5)
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])
