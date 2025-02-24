from .frame import Frame
import gtsam
import numpy as np
from typing import List, Dict, Set

class Keyframe(Frame):
    _next_id = 0
    
    def __init__(self, frame: Frame):
        super().__init__(frame.image, frame.depth, frame.camera_matrix)
        self.id = Keyframe._next_id
        Keyframe._next_id += 1
        
        # Copy frame data
        self.pose = frame.pose.copy() if frame.pose is not None else None
        self.keypoints = frame.keypoints.copy() if frame.keypoints is not None else None
        self.descriptors = frame.descriptors.copy() if frame.descriptors is not None else None
        self.gt_pose = frame.gt_pose.copy() if frame.gt_pose is not None else None
        self.timestamp = frame.timestamp
        
        # Additional keyframe specific data
        self.visible_map_points: Set[int] = set()
        self.connected_keyframes: Set[int] = set()
        self.factors: List[gtsam.NonlinearFactor] = []
        
    @property
    def pose_key(self) -> int:
        """Get pose key for GTSAM optimization."""
        return gtsam.symbol('x', self.id)
        
    def add_odometry_factor(self, prev_keyframe, 
                           noise_model: gtsam.noiseModel.Base = None):
        if noise_model is None:
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]))  # x,y,z,roll,pitch,yaw
        
        # Ensure we compute relative transform as inverse(prev) * current
        relative_pose = gtsam.Pose3(
            prev_keyframe.pose_inverse() @ self.pose)
        
        factor = gtsam.BetweenFactorPose3(
            prev_keyframe.pose_key, 
            self.pose_key,
            relative_pose,
            noise_model)
        
        self.factors.append(factor)
        
    def add_loop_closure_factor(self, loop_keyframe, relative_pose: np.ndarray,
                              noise_model: gtsam.noiseModel.Base = None):
        if noise_model is None:
            # More uncertainty in loop closures
            noise_model = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.2, 0.2, 0.2, 0.4, 0.4, 0.4]))
            
        factor = gtsam.BetweenFactorPose3(
            loop_keyframe.pose_key,
            self.pose_key,
            gtsam.Pose3(relative_pose),
            noise_model)
        
        self.factors.append(factor)
        
    def __str__(self) -> str:
        status = [super().__str__()]  # Get Frame info first
        
        status.append("\nKeyframe-specific:")
        status.append(f"Reference frame: {self.reference_frame_id}")
        status.append(f"Local gaussians: {len(self.local_gaussians)}")
        status.append(f"Loop candidates: {len(self.loop_closure_candidates)}")
        status.append(f"GTSAM factors: {len(self.factors)}")
        
        return "\n".join(status)
