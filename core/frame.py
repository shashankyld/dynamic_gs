import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class FrameType(Enum):
    FRAME = 0
    KEYFRAME = 1

@dataclass
class FeatureTrack:
    keypoint: np.ndarray  # 2D point
    descriptor: np.ndarray
    depth: float
    is_dynamic: bool = False
    track_id: int = -1

@dataclass
class MotionModel:
    """Constant velocity motion model"""
    linear_velocity: np.ndarray  # 3x1 vector
    angular_velocity: np.ndarray  # 3x1 vector
    last_update: float  # timestamp
    
    def predict_pose(self, current_time: float, last_pose: np.ndarray) -> np.ndarray:
        dt = current_time - self.last_update
        # Simple constant velocity prediction
        translation = last_pose[:3, 3] + self.linear_velocity * dt
        # Use exponential map for rotation prediction
        angle = np.linalg.norm(self.angular_velocity) * dt
        if angle > 0:
            axis = self.angular_velocity / np.linalg.norm(self.angular_velocity)
            R = self.exponential_map(axis, angle) @ last_pose[:3, :3]
        else:
            R = last_pose[:3, :3]
            
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = translation
        return pose
    
    @staticmethod
    def exponential_map(axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to rotation matrix"""
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        return R

class Frame:
    def __init__(self, 
                 frame_id: int,
                 timestamp: float,
                 image: Optional[np.ndarray] = None,
                 depth: Optional[np.ndarray] = None,
                 dynamic_mask: Optional[np.ndarray] = None,
                 gt_pose: Optional[np.ndarray] = None,
                 camera_matrix: Optional[np.ndarray] = None):
        
        self.id = frame_id
        self.timestamp = timestamp
        self.frame_type = FrameType.FRAME
        
        # Images and masks
        self._image = image
        self._depth = depth
        self._dynamic_mask = dynamic_mask
        
        # Pose information
        self.gt_pose = gt_pose if gt_pose is not None else np.eye(4)
        self.pose = None  # Estimated pose
        self.motion_model = MotionModel(
            np.zeros(3), np.zeros(3), timestamp)
        
        # Features
        self.keypoints = None  # Nx2 array
        self.descriptors = None  # NxD array
        self.dynamic_kp_indices = set()  # Indices of keypoints on dynamic objects
        
        # Camera parameters
        self.camera_matrix = camera_matrix
        if camera_matrix is not None:
            self.fx = camera_matrix[0, 0]
            self.fy = camera_matrix[1, 1]
            self.cx = camera_matrix[0, 2]
            self.cy = camera_matrix[1, 2]
            
        # Visibility information
        self.visible_map_points = set()  # Map points visible in this frame
        
    def set_features(self, keypoints: np.ndarray, descriptors: np.ndarray):
        """Set feature keypoints and descriptors"""
        self.keypoints = keypoints
        self.descriptors = descriptors
        
    def mark_dynamic_keypoints(self, indices: Set[int]):
        """Mark keypoints as belonging to dynamic objects"""
        self.dynamic_kp_indices.update(indices)
        
    def is_keypoint_dynamic(self, kp_idx: int) -> bool:
        """Check if a keypoint is on a dynamic object"""
        return kp_idx in self.dynamic_kp_indices
    
    def clear_images(self, keep_for_gs: bool = False):
        """Clear image data to save memory"""
        if not keep_for_gs or self.frame_type == FrameType.FRAME:
            self._image = None
            self._depth = None
            self._dynamic_mask = None
            
    def get_visible_map_points(self, map_points: Dict[int, np.ndarray], 
                              frustum_margin: float = 0.1) -> Set[int]:
        """Get map points that should be visible in this frame"""
        if self.camera_matrix is None:
            return set()
            
        visible_points = set()
        for point_id, point_3d in map_points.items():
            # Transform point to camera frame
            point_cam = self.pose_inverse() @ np.append(point_3d, 1)
            
            # Check if point is in front of camera
            if point_cam[2] <= 0:
                continue
                
            # Project to image plane
            point_2d = self.camera_matrix @ point_cam[:3]
            point_2d = point_2d / point_2d[2]
            
            # Check if point projects within image bounds (with margin)
            if self._image is not None:
                h, w = self._image.shape[:2]
                margin_x = w * frustum_margin
                margin_y = h * frustum_margin
                if (-margin_x <= point_2d[0] <= w + margin_x and
                    -margin_y <= point_2d[1] <= h + margin_y):
                    visible_points.add(point_id)
                    
        self.visible_map_points = visible_points
        return visible_points
    
    def pose_inverse(self) -> np.ndarray:
        """Get inverse of current pose"""
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = self.pose[:3, :3].T
        inv_pose[:3, 3] = -self.pose[:3, :3].T @ self.pose[:3, 3]
        return inv_pose
    
    @property
    def image(self) -> Optional[np.ndarray]:
        return self._image
        
    @property
    def depth(self) -> Optional[np.ndarray]:
        return self._depth
        
    @property
    def dynamic_mask(self) -> Optional[np.ndarray]:
        return self._dynamic_mask
