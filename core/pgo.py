import gtsam
import numpy as np
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

@dataclass
class OptimizationParams:
    """Parameters for pose graph optimization"""
    max_iterations: int = 100
    relative_error_threshold: float = 1e-5
    absolute_error_threshold: float = 1e-5
    verbose: bool = False

class PoseGraphOptimizer:
    def __init__(self, mode: str = "batch"):
        # Initialize base objects
        self.mode = mode
        self.graph_factors = gtsam.NonlinearFactorGraph()
        self.graph_initials = gtsam.Values()
        
        # Noise models with better defaults based on working implementation
        self.fixed_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9]))
            
        # Standard deviation for translation (m) and rotation (degrees)
        tran_std = 0.1  # meters
        rot_std = 2.0   # degrees
        
        # Convert degrees to radians for rotation noise
        self.const_noise = np.array([
            np.radians(rot_std), np.radians(rot_std), np.radians(rot_std),
            tran_std, tran_std, tran_std])
            
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(self.const_noise)
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(self.const_noise)
        
        # ISAM2 specific setup
        if mode == "incremental":
            self.isam = gtsam.ISAM2()
        else:
            self.isam = None
            
        self.current_pose = None
        self.current_node_idx = None
        self.optimized_values = None
            
    def add_pose_node(self, pose_id: int, pose: np.ndarray, is_first: bool = False):
        """Add a new pose node with optional prior"""
        pose_key = gtsam.symbol('x', pose_id)
        gtsam_pose = gtsam.Pose3(pose)
        
        # Add to initial values
        if not self.graph_initials.exists(pose_key):
            self.graph_initials.insert(pose_key, gtsam_pose)
            
        # Add prior factor for first pose
        if is_first:
            self.graph_factors.add(
                gtsam.PriorFactorPose3(pose_key, gtsam_pose, self.fixed_noise))
                
    def add_odometry_edge(self, pose_id1: int, pose_id2: int, relative_pose: np.ndarray):
        """Add odometry constraint between consecutive poses"""
        pose1_key = gtsam.symbol('x', pose_id1)
        pose2_key = gtsam.symbol('x', pose_id2)
        
        # Create odometry factor
        factor = gtsam.BetweenFactorPose3(
            pose1_key, pose2_key, 
            gtsam.Pose3(relative_pose), 
            self.odometry_noise)
            
        self.graph_factors.add(factor)
        
    def add_loop_closure(self, pose_id1: int, pose_id2: int, relative_pose: np.ndarray):
        """Add loop closure constraint"""
        pose1_key = gtsam.symbol('x', pose_id1)
        pose2_key = gtsam.symbol('x', pose_id2)
        
        # Create loop closure factor
        factor = gtsam.BetweenFactorPose3(
            pose1_key, pose2_key,
            gtsam.Pose3(relative_pose),
            self.loop_noise)
            
        self.graph_factors.add(factor)
            
    def optimize(self) -> Dict[int, np.ndarray]:
        """Optimize the pose graph"""
        try:
            if self.mode == "incremental":
                # ISAM2 optimization
                self.isam.update(self.graph_factors, self.graph_initials)
                result = self.isam.calculateEstimate()
                
                # Reset graph for next iteration in ISAM2 mode
                self.graph_factors = gtsam.NonlinearFactorGraph()
                self.graph_initials.clear()
                
            else:
                # Batch optimization
                optimizer = gtsam.LevenbergMarquardtOptimizer(
                    self.graph_factors, self.graph_initials)
                result = optimizer.optimize()
                
                # Update initial values for next optimization
                self.graph_initials = result
                
            # Extract optimized poses
            optimized_poses = {}
            for key in result.keys():
                pose_id = int(gtsam.Symbol(key).string()[1:])
                optimized_poses[pose_id] = result.atPose3(key).matrix()
                
            return optimized_poses
            
        except RuntimeError as e:
            logging.error(f"Optimization failed: {str(e)}")
            return {}
    
    def write_g2o(self, filename: str):
        """Save pose graph in g2o format"""
        gtsam.writeG2o(self.graph_factors, self.graph_initials, filename)
