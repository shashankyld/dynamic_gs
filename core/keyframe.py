from .frame import Frame
import gtsam
import numpy as np
from typing import List, Dict, Set

class Keyframe(Frame):
    def __init__(self, frame: Frame):
        super().__init__(frame.id, frame.timestamp, frame.image, frame.depth)
        self.__dict__.update(frame.__dict__)  # Copy all frame attributes
        
        # Additional Keyframe-specific attributes
        self.local_gaussians: List[int] = []  # Indices of associated gaussians
        self.reference_frame_id = -1
        self.loop_closure_candidates: List[int] = []
        
        # Factor graph related
        self.pose_key = gtsam.symbol('x', self.id)
        self.factors: List[gtsam.NonlinearFactor] = []
        
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
