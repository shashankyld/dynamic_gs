import numpy as np
from core.pgo import PoseGraphOptimizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_rotation_matrix(yaw, pitch, roll):
    """Create 3x3 rotation matrix from Euler angles (in radians)"""
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    return Rz @ Ry @ Rx

def create_se3(translation, rotation_euler):
    """Create SE(3) matrix from translation and Euler angles (in degrees)"""
    rot_mat = create_rotation_matrix(*np.radians(rotation_euler))
    pose = np.eye(4)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = translation
    return pose

def plot_poses(poses_dict, ax, style='b-', marker='o', label=None):
    """Plot poses as a trajectory with coordinate frames"""
    positions = np.array([pose[:3, 3] for pose in poses_dict.values()])
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], style, label=label)
    
    # Plot pose coordinate frames
    for pose_id, pose in poses_dict.items():
        # Position
        pos = pose[:3, 3]
        
        # Rotation axes (scaled down)
        scale = 0.2
        x_axis = pos + scale * pose[:3, 0]
        y_axis = pos + scale * pose[:3, 1]
        z_axis = pos + scale * pose[:3, 2]
        
        # Plot coordinate frame
        ax.plot([pos[0], x_axis[0]], [pos[1], x_axis[1]], [pos[2], x_axis[2]], 'r-', linewidth=1)
        ax.plot([pos[0], y_axis[0]], [pos[1], y_axis[1]], [pos[2], y_axis[2]], 'g-', linewidth=1)
        ax.plot([pos[0], z_axis[0]], [pos[1], z_axis[1]], [pos[2], z_axis[2]], 'b-', linewidth=1)
        
        # Add pose ID label
        ax.text(pos[0], pos[1], pos[2], f' {pose_id}', fontsize=8)
    
    # Plot markers at pose positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='black', marker=marker)

def test_pose_graph():
    print("Testing Pose Graph Optimization...")
    
    # Initialize optimizer
    pgo = PoseGraphOptimizer(mode="batch")
    
    try:
        # Create more complex test scenario
        poses = {
            # Origin
            1: np.eye(4),
            
            # Moving forward with slight rotation
            2: create_se3([1.0, 0.1, 0.0], [5, 0, 0]),
            
            # Sharp right turn
            3: create_se3([1.5, 1.0, 0.0], [45, 0, 0]),
            
            # Moving up incline
            4: create_se3([1.2, 2.0, 0.5], [45, 15, 0]),
            
            # Complex motion
            5: create_se3([0.8, 2.5, 1.0], [30, 20, 10]),
            
            # Loop closure - should be near start but with noise
            6: create_se3([0.2, 0.3, 0.1], [5, 5, 5])
        }
        
        # Create figure for visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Plot ground truth
        ax1 = fig.add_subplot(131, projection='3d')
        plot_poses(poses, ax1, 'k-', 'o', label='Ground Truth')
        ax1.set_title('Ground Truth Trajectory')
        ax1.legend()
        
        # Add poses with increasing uncertainty
        for idx, pose in poses.items():
            pgo.add_pose_node(idx, pose, is_first=(idx==1))
        
        # Add noisy odometry edges with visualization
        noisy_poses = {1: poses[1]}  # Start with first pose
        current_pose = poses[1]
        
        for i in range(1, len(poses)):
            rel_pose = np.linalg.inv(poses[i]) @ poses[i+1]
            # Add noise to odometry
            noise_t = np.random.normal(0, 0.05, 3)  # 5cm std noise
            noise_r = np.random.normal(0, np.radians(2), 3)  # 2 degree std noise
            rel_pose[:3, 3] += noise_t
            rel_pose[:3, :3] = create_rotation_matrix(*noise_r) @ rel_pose[:3, :3]
            
            # Accumulate noisy pose
            current_pose = current_pose @ rel_pose
            noisy_poses[i+1] = current_pose
            
            # Add to optimizer
            pgo.add_odometry_edge(i, i+1, rel_pose)
        
        # Plot noisy trajectory
        ax2 = fig.add_subplot(132, projection='3d')
        plot_poses(noisy_poses, ax2, 'r--', 's', label='Noisy Trajectory')
        ax2.set_title('Before Optimization')
        ax2.legend()
        
        # Add loop closure with different levels of uncertainty
        # Strong loop closure - start to end
        pgo.add_loop_closure(1, 6, np.linalg.inv(poses[1]) @ poses[6])
        
        # Weak loop closure - middle links
        mid_rel_pose = np.linalg.inv(poses[2]) @ poses[5]
        mid_rel_pose[:3, 3] += np.random.normal(0, 0.1, 3)  # More noise
        pgo.add_loop_closure(2, 5, mid_rel_pose)
        
        # Optimize
        optimized_poses = pgo.optimize()
        
        if not optimized_poses:
            print("Error: Optimization returned no poses!")
            return False
        
        # Plot optimized trajectory
        ax3 = fig.add_subplot(133, projection='3d')
        plot_poses(optimized_poses, ax3, 'g-', '^', label='Optimized Trajectory')
        ax3.set_title('After Optimization')
        ax3.legend()
        
        # Set consistent axes limits
        xlim = ylim = zlim = None
        for ax in [ax1, ax2, ax3]:
            if xlim is None:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
            else:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()
        
        # Analyze results
        print("\nOptimization Results:")
        position_errors = []
        rotation_errors = []
        
        for pose_id, pose_mat in optimized_poses.items():
            gt_pose = poses[pose_id]
            
            # Position error
            pos_error = np.linalg.norm(pose_mat[:3, 3] - gt_pose[:3, 3])
            position_errors.append(pos_error)
            
            # Rotation error (in degrees)
            R_error = np.arccos(
                (np.trace(pose_mat[:3, :3] @ gt_pose[:3, :3].T) - 1) / 2)
            rotation_errors.append(np.degrees(R_error))
            
            print(f"\nPose {pose_id}:")
            print(f"Position: [{pose_mat[0,3]:.3f}, {pose_mat[1,3]:.3f}, {pose_mat[2,3]:.3f}]")
            print(f"Position Error: {pos_error:.3f}m")
            print(f"Rotation Error: {np.degrees(R_error):.1f}°")
        
        print(f"\nSummary Statistics:")
        print(f"Mean Position Error: {np.mean(position_errors):.3f}m")
        print(f"Max Position Error: {np.max(position_errors):.3f}m")
        print(f"Mean Rotation Error: {np.mean(rotation_errors):.1f}°")
        print(f"Max Rotation Error: {np.max(rotation_errors):.1f}°")
        
        # Test passes if errors are within bounds
        success = (np.max(position_errors) < 0.2 and  # 20cm max error
                  np.max(rotation_errors) < 5.0)      # 5 degrees max error
        
        if success:
            print("\nPose Graph test successful!")
        else:
            print("\nPose Graph test failed - errors too large!")
            
        return success
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_pose_graph()
    if not success:
        print("\nTest FAILED")
