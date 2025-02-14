import open3d as o3d
import numpy as np

def create_camera_trajectory(poses):
    """Create trajectory visualization from camera poses."""
    points = [pose[:3, 3] for pose in poses.values()]
    if not points:
        return None, None

    # Create point cloud for trajectory points
    trajectory = o3d.geometry.PointCloud()
    trajectory.points = o3d.utility.Vector3dVector(np.array(points))
    trajectory.paint_uniform_color([1, 0, 0])  # Red color for points

    # Create line set for connecting trajectory points
    lines = [[i, i+1] for i in range(len(points)-1)]
    if lines:
        line_set = o3d.geometry.LineSet()
        line_set.points = trajectory.points
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color([0, 1, 0])  # Green color for lines
        return trajectory, line_set
    
    return trajectory, None

def create_camera_frames(poses, size=0.1):
    """Create coordinate frames for each camera pose."""
    frames = []
    for pose in poses.values():
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        cam_frame.transform(pose)
        frames.append(cam_frame)
    return frames

def visualize_map_and_trajectory(global_map, global_poses, voxel_size=0.05):
    """Visualize the global map with camera trajectory and poses."""
    # Downsample the global map
    if voxel_size > 0:
        global_map = global_map.voxel_down_sample(voxel_size=voxel_size)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(world_frame)

    # Add global map
    vis.add_geometry(global_map)

    # Add camera frames
    for frame in create_camera_frames(global_poses):
        vis.add_geometry(frame)

    # Add trajectory visualization
    trajectory, line_set = create_camera_trajectory(global_poses)
    if trajectory is not None:
        vis.add_geometry(trajectory)
    if line_set is not None:
        vis.add_geometry(line_set)

    # Set default view
    vis.get_view_control().set_zoom(0.5)
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, -1, 0])

    # Run visualizer
    vis.run()
    vis.destroy_window()
