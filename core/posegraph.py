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
        
        # Noise models for different constraints
        self.between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))  # More relaxed than prior
            
    def add_keyframe(self, keyframe: Keyframe, add_prior: bool = False):
        """Add keyframe to pose graph with between factors to previous keyframe."""
        self.keyframes[keyframe.id] = keyframe
        
        # Add pose to initial estimates
        pose3 = gtsam.Pose3(keyframe.pose)
        self.initial_estimates.insert(keyframe.pose_key, pose3)
        
        # Add prior factor for first keyframe
        if add_prior or len(self.keyframes) == 1:
            prior_factor = gtsam.PriorFactorPose3(
                keyframe.pose_key, pose3, self.prior_noise)
            self.graph.add(prior_factor)
            print(f"Added prior factor to keyframe {keyframe.id}")
            
        # Add between factor with previous keyframe
        else:
            # Find previous keyframe
            prev_keyframe_id = max([k for k in self.keyframes.keys() if k < keyframe.id])
            if prev_keyframe_id in self.keyframes:
                prev_keyframe = self.keyframes[prev_keyframe_id]
                
                # Compute relative transform between keyframes
                relative_pose = np.linalg.inv(prev_keyframe.pose) @ keyframe.pose
                between_pose = gtsam.Pose3(relative_pose)
                
                # Add between factor
                between_factor = gtsam.BetweenFactorPose3(
                    prev_keyframe.pose_key,
                    keyframe.pose_key,
                    between_pose,
                    self.between_noise)
                self.graph.add(between_factor)
                print(f"Added between factor from keyframe {prev_keyframe_id} to {keyframe.id}")

    def optimize(self, max_iterations: int = 100) -> bool:
        """Optimize pose graph with more debug info."""
        try:
            print(f"Optimizing pose graph with {len(self.keyframes)} keyframes")
            print(f"Graph size: {self.graph.size()} factors")
            print(f"Initial values size: {self.initial_estimates.size()} values")
            
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(max_iterations)
            params.setVerbosity('ERROR')
            
            optimizer = gtsam.LevenbergMarquardtOptimizer(
                self.graph, self.initial_estimates, params)
            result = optimizer.optimize()
            
            # Update keyframe poses
            for kf in self.keyframes.values():
                if result.exists(kf.pose_key):  # Check if pose exists in results
                    new_pose = result.atPose3(kf.pose_key)
                    kf.pose = new_pose.matrix()
                else:
                    print(f"Warning: No result for keyframe {kf.id}")
                    
            # Update initial estimates for next optimization
            self.initial_estimates = result
            return True
            
        except RuntimeError as e:
            print(f"Optimization failed: {e}")
            return False
