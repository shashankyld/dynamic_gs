import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from core.posegraph import PoseGraph
from core.frame import Frame
from core.keyframe import Keyframe
import gtsam

def create_test_frame(frame_id: int, timestamp: float, pose: np.ndarray) -> Frame:
    """Create a test frame with dummy image data"""
    # Create dummy image and depth (16x16 for testing)
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    depth = np.ones((16, 16), dtype=np.float32)
    
    # Create camera matrix
    camera_matrix = np.array([
        [500, 0, 8],
        [0, 500, 8],
        [0, 0, 1]
    ])
    
    frame = Frame(
        frame_id=frame_id,
        timestamp=timestamp,
        image=image,
        depth=depth,
        dynamic_mask=None,
        gt_pose=pose,
        camera_matrix=camera_matrix
    )
    
    # Add some dummy features
    frame.keypoints = np.random.rand(10, 2) * 16  # 10 random keypoints
    frame.descriptors = np.random.rand(10, 128)    # 128D descriptors
    
    return frame

def create_trajectory_poses():
    """Create a simpler circular trajectory"""
    poses = {}
    timestamps = {}
    n_poses = 20  # More poses for better optimization
    radius = 1.0  # Smaller radius
    
    for i in range(n_poses):
        angle = 2 * np.pi * i / (n_poses - 1)
        # Create simpler planar motion
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0.0  # Keep it planar for simplicity
        
        pose = np.eye(4)
        pose[:3, 3] = [x, y, z]
        
        # Make camera orientation more stable
        # Always face center with consistent up direction
        position = np.array([x, y, z])
        forward = -position / np.linalg.norm(position + 1e-10)  # Avoid division by zero
        up = np.array([0.0, 0.0, 1.0])  # Consistent up vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right + 1e-10)
        up = np.cross(forward, right)  # Recompute up to ensure orthogonality
        
        R = np.column_stack([forward, right, up])
        pose[:3, :3] = R
        
        poses[i] = pose
        timestamps[i] = i * 0.5
        
    return poses, timestamps

def plot_poses_and_points(ax, poses, keyframes, title):
    """Plot poses and visible map points"""
    # Plot trajectory
    positions = np.array([pose[:3, 3] for pose in poses.values()])
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Trajectory')
    
    # Plot camera poses
    for pose_id, pose in poses.items():
        pos = pose[:3, 3]
        
        # Camera axes
        axis_length = 0.3
        x_axis = pos + axis_length * pose[:3, 0]  # Forward (red)
        y_axis = pos + axis_length * pose[:3, 1]  # Right (green)
        z_axis = pos + axis_length * pose[:3, 2]  # Up (blue)
        
        ax.plot([pos[0], x_axis[0]], [pos[1], x_axis[1]], [pos[2], x_axis[2]], 'r-', linewidth=1)
        ax.plot([pos[0], y_axis[0]], [pos[1], y_axis[1]], [pos[2], y_axis[2]], 'g-', linewidth=1)
        ax.plot([pos[0], z_axis[0]], [pos[1], z_axis[1]], [pos[2], z_axis[2]], 'b-', linewidth=1)
        
        # Pose ID
        ax.text(pos[0], pos[1], pos[2], f' {pose_id}', fontsize=8)
    
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='black', marker='o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

def add_realistic_noise(pose: np.ndarray, position_std: float = 0.05, rotation_std: float = 1.0):
    """Add realistic noise to pose"""
    noisy_pose = pose.copy()
    
    # Add position noise
    noisy_pose[:3, 3] += np.random.normal(0, position_std, 3)
    
    # Add rotation noise (in degrees, converted to radians)
    angle_noise = np.random.normal(0, np.radians(rotation_std), 3)
    
    # Convert angle-axis to rotation matrix
    angle = np.linalg.norm(angle_noise)
    if angle > 0:
        axis = angle_noise / angle
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        R_noise = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        noisy_pose[:3, :3] = R_noise @ pose[:3, :3]
    
    return noisy_pose

def test_posegraph():
    print("Testing PoseGraph with Keyframes...")
    
    # Create trajectory
    poses, timestamps = create_trajectory_poses()
    
    # Create visualization figure
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Plot ground truth
    plot_poses_and_points(ax1, poses, None, 'Ground Truth')
    
    # Initialize pose graph with better noise models
    pose_graph = PoseGraph()
    
    try:
        noisy_poses = {}
        prev_noisy_pose = None
        
        for idx, (pose, timestamp) in enumerate(zip(poses.values(), timestamps.values())):
            # Add consistent noise
            if idx == 0:
                noisy_pose = add_realistic_noise(pose, 0.01, 0.5)
                prev_noisy_pose = noisy_pose
            else:
                relative_pose = np.linalg.inv(poses[idx-1]) @ pose
                noisy_relative = add_realistic_noise(relative_pose, 0.02, 1.0)
                noisy_pose = prev_noisy_pose @ noisy_relative
                prev_noisy_pose = noisy_pose
                
            noisy_poses[idx] = noisy_pose
            
            frame = create_test_frame(idx, timestamp, noisy_pose)
            keyframe = Keyframe(frame)
            pose_graph.add_keyframe(keyframe, add_prior=(idx==0))
            
            if idx > 0:
                prev_keyframe = pose_graph.keyframes[idx-1]
                # Compute relative pose correctly:
                relative_pose = np.linalg.inv(prev_keyframe.pose) @ keyframe.pose
                # Instead of multiplying by displacement, use a constant noise level:
                odom_noise = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
                noise_model = gtsam.noiseModel.Diagonal.Sigmas(odom_noise)
                keyframe.add_odometry_factor(prev_keyframe, noise_model)
            
            if idx == len(poses)-1:
                first_keyframe = pose_graph.keyframes[0]
                relative_pose = np.linalg.inv(first_keyframe.pose) @ keyframe.pose
                # For loop closures, use a slightly looser (higher) noise sigma:
                loop_noise = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
                loop_noise_model = gtsam.noiseModel.Diagonal.Sigmas(loop_noise)
                keyframe.add_loop_closure_factor(first_keyframe, relative_pose, loop_noise_model)
        
        # Plot noisy trajectory
        plot_poses_and_points(ax2, noisy_poses, None, 'Before Optimization')
        
        success = pose_graph.optimize()
        
        if success:
            optimized_poses = {kf.id: kf.pose for kf in pose_graph.keyframes.values()}
            plot_poses_and_points(ax3, optimized_poses, pose_graph.keyframes, 'After Optimization')
            
            position_errors = []
            rotation_errors = []
            
            for pose_id, opt_pose in optimized_poses.items():
                gt_pose = poses[pose_id]
                pos_error = np.linalg.norm(opt_pose[:3, 3] - gt_pose[:3, 3])
                position_errors.append(pos_error)
                R_error = np.arccos((np.trace(opt_pose[:3, :3] @ gt_pose[:3, :3].T) - 1) / 2)
                rotation_errors.append(np.degrees(R_error))
                print(f"\nKeyframe {pose_id}:")
                print(f"Position Error: {pos_error:.3f}m")
                print(f"Rotation Error: {np.degrees(R_error):.1f}°")
            
            print(f"\nSummary Statistics:")
            print(f"Mean Position Error: {np.mean(position_errors):.3f}m")
            print(f"Max Position Error: {np.max(position_errors):.3f}m")
            print(f"Mean Rotation Error: {np.mean(rotation_errors):.1f}°")
            print(f"Max Rotation Error: {np.max(rotation_errors):.1f}°")
            
            plt.tight_layout()
            plt.show()
            
            return np.max(position_errors) < 0.2
            
        else:
            print("Optimization failed!")
            return False
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_posegraph()
    if not success:
        print("\nTest FAILED")
