import gtsam
import numpy as np
from typing import Dict, List, Optional
from .keyframe import Keyframe

class PoseGraph:
    def __init__(self):
        self.keyframes: Dict[int, Keyframe] = {}
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        
        # Optimization parameters
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimates)
        
        # Prior noise model
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
        
    def add_keyframe(self, keyframe: Keyframe, add_prior: bool = False):
        self.keyframes[keyframe.id] = keyframe
        
        # Add pose to initial estimates
        pose3 = gtsam.Pose3(keyframe.pose)
        self.initial_estimates.insert(keyframe.pose_key, pose3)
        
        # Add prior factor for first keyframe
        if add_prior or len(self.keyframes) == 1:
            prior_factor = gtsam.PriorFactorPose3(
                keyframe.pose_key, pose3, self.prior_noise)
            self.graph.add(prior_factor)
        
        # Add all factors from keyframe
        for factor in keyframe.factors:
            self.graph.add(factor)
            
    def optimize(self, max_iterations: int = 100) -> bool:
        try:
            self.optimizer = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.initial_estimates)
            result = self.optimizer.optimize()
            
            # Update keyframe poses
            for kf in self.keyframes.values():
                new_pose = result.atPose3(kf.pose_key)
                kf.pose = new_pose.matrix()
                
            # Update initial estimates
            self.initial_estimates = result
            return True
            
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            return False
