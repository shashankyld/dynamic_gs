import numpy as np
from typing import Dict, Optional, List
from core.frame import Frame, FrameType
from core.keyframe import Keyframe
from core.track import Tracker
from core.map import Map
from collections import deque
import torch
import time
from slam_parameters import SlamParameters
import math
from core.posegraph import PoseGraph

class SLAMSystem:
    """Core SLAM system managing tracking, mapping and frame management."""
    
    def __init__(self, camera_matrix: np.ndarray, 
                 num_features: int = SlamParameters.NUM_FEATURES,
                 num_local_keyframes: int = SlamParameters.NUM_LOCAL_KEYFRAMES,
                 device: Optional[torch.device] = None):
        """Initialize SLAM system."""
        self.camera_matrix = camera_matrix
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Core components
        self.tracker = Tracker(camera_matrix, device=self.device, num_features=num_features)
        self.map = Map()
        
        # Parameters
        self.num_local_keyframes = num_local_keyframes
        self.min_keyframe_matches = 100  # Minimum matches to consider before creating new keyframe
        self.min_inlier_threshold = 30  # Minimum inliers for good tracking
        
        # State
        self.initialized = False
        self.current_frame: Optional[Frame] = None
        self.last_keyframe: Optional[Keyframe] = None
        self.latest_pose = np.eye(4)
        self.local_keyframes = deque(maxlen=num_local_keyframes)
        self.prev_frame = None  # Store previous frame
        
        # Additional tracking state
        self.frames_since_keyframe = 0
        self.distance_since_keyframe = 0.0
        self.rotation_since_keyframe = 0.0
        
        # Add pose graph
        self.pose_graph = PoseGraph()
        
    def initialize(self, frame: Frame) -> bool:
        """Initialize SLAM with first frame."""
        if self.initialized:
            return False
            
        # Ensure frame has features
        if not self.tracker.extract_features(frame):
            return False
            
        # Convert first frame to keyframe
        keyframe = Keyframe(frame)
        keyframe.pose = np.eye(4)  # Set as origin
        
        # Add to map
        self.map.add_keyframe(keyframe)
        self.last_keyframe = keyframe
        self.local_keyframes.append(keyframe)
        
        self.initialized = True
        return True
        
    def track_frame(self, frame: Frame, dynamic_mask: Optional[np.ndarray] = None) -> bool:
        """Process new frame with frame-to-frame tracking, fallback to keyframe tracking if needed."""
        start_time = time.time()
        if not self.initialized:
            return self.initialize(frame)
            
        if not self.tracker.extract_features(frame):
            return False
            
        self.current_frame = frame
        tracking_status = "frame-to-frame"
        inliers = None
        success = False
        
        # Try frame-to-frame tracking first
        if self.prev_frame is not None:
            if dynamic_mask is not None:
                rel_pose, inliers = self.tracker.track_frames_with_mask(
                    self.prev_frame, frame, dynamic_mask)
            else:
                rel_pose, inliers = self.tracker.track_frames(self.prev_frame, frame)
                
            if rel_pose is not None:
                num_inliers = len(inliers) if inliers is not None else 0
                if num_inliers >= self.min_inlier_threshold:
                    # Good frame-to-frame tracking
                    frame.pose = self.prev_frame.pose @ rel_pose
                    print(f"[Tracking] Good frame-to-frame tracking with {num_inliers} inliers")
                    success = True
                else:
                    # Try keyframe tracking as fallback
                    print(f"[Warning] Poor frame-to-frame tracking ({num_inliers} inliers), trying keyframe")
                    tracking_status = "keyframe"
                    if dynamic_mask is not None:
                        rel_pose, inliers = self.tracker.track_frames_with_mask(
                            self.last_keyframe, frame, dynamic_mask)
                    else:
                        rel_pose, inliers = self.tracker.track_frames(self.last_keyframe, frame)
                    
                    if rel_pose is not None:
                        num_inliers = len(inliers) if inliers is not None else 0
                        if num_inliers >= self.min_inlier_threshold:
                            frame.pose = self.last_keyframe.pose @ rel_pose
                            print(f"[Tracking] Recovered with keyframe tracking ({num_inliers} inliers)")
                            success = True
                        else:
                            print(f"[Warning] Poor tracking even with keyframe ({num_inliers} inliers)")
                            return False
                    else:
                        print("[Error] Failed to track against keyframe")
                        return False
            else:
                print("[Error] Failed frame-to-frame tracking")
                return False
        else:
            # First frame after initialization - track against keyframe
            if dynamic_mask is not None:
                rel_pose, inliers = self.tracker.track_frames_with_mask(
                    self.last_keyframe, frame, dynamic_mask)
            else:
                rel_pose, inliers = self.tracker.track_frames(self.last_keyframe, frame)
                
            if rel_pose is not None:
                frame.pose = self.last_keyframe.pose @ rel_pose
                success = True
            else:
                print("[Error] Failed to track first frame against keyframe")
                return False
                
        # Check if we should create new keyframe
        if success and self._need_new_keyframe(frame, len(inliers) if inliers is not None else 0):
            keyframe = self._create_keyframe(frame)
            self._update_local_map(keyframe)
            print(f"[Mapping] Created new keyframe (ID: {keyframe.id})")
            
        # Update previous frame
        self.prev_frame = frame
        print(f"[Timing] Frame tracking took {time.time() - start_time:.2f} seconds")
        return success
        
    def track_frame_to_keyframe(self, frame: Frame, dynamic_mask: Optional[np.ndarray] = None) -> bool:
        """Process new frame by always tracking against the last keyframe."""
        start_time = time.time()
        if not self.initialized:
            return self.initialize(frame)
            
        if not self.tracker.extract_features(frame):
            return False
            
        self.current_frame = frame
        inliers = None
        success = False
        
        # Always track against last keyframe
        if dynamic_mask is not None:
            rel_pose, inliers = self.tracker.track_frames_with_mask(
                self.last_keyframe, frame, dynamic_mask)
        else:
            rel_pose, inliers = self.tracker.track_frames(self.last_keyframe, frame)
            
        if rel_pose is not None:
            num_inliers = len(inliers) if inliers is not None else 0
            if num_inliers >= self.min_inlier_threshold:
                frame.pose = self.last_keyframe.pose @ rel_pose
                print(f"[Tracking] Good keyframe tracking with {num_inliers} inliers")
                success = True
                
                # Check if we should create new keyframe
                if self._need_new_keyframe(frame, num_inliers):
                    keyframe = self._create_keyframe(frame)
                    self._update_local_map(keyframe)
                    print(f"[Mapping] Created new keyframe (ID: {keyframe.id})")
            else:
                print(f"[Warning] Poor tracking against keyframe ({num_inliers} inliers)")
                return False
        else:
            print("[Error] Failed to track against keyframe")
            return False
                
        # Update previous frame (still maintain this for visualization purposes)
        self.prev_frame = frame
        print(f"[Timing] Keyframe tracking took {time.time() - start_time:.2f} seconds")
        return success

    def _need_new_keyframe(self, frame: Frame, num_matches: int) -> bool:
        """Enhanced keyframe decision based on multiple criteria."""
        self.frames_since_keyframe += 1
        
        # Calculate motion since last keyframe
        if self.last_keyframe is not None and frame.pose is not None:
            # Get relative transform from last keyframe to current frame
            rel_transform = np.linalg.inv(self.last_keyframe.pose) @ frame.pose
            
            # Calculate translation distance
            translation = np.linalg.norm(rel_transform[:3, 3])
            self.distance_since_keyframe += translation
            
            # Calculate rotation angle (in degrees)
            R = rel_transform[:3, :3]
            trace = np.trace(R)
            angle = np.rad2deg(np.arccos((trace - 1) / 2))
            self.rotation_since_keyframe += abs(angle)
            
            # Debug info
            print(f"[Keyframe Check] Distance: {self.distance_since_keyframe:.2f}m, "
                  f"Rotation: {self.rotation_since_keyframe:.1f}°, "
                  f"Frames: {self.frames_since_keyframe}")
            
            # Check criteria
            if (self.distance_since_keyframe >= SlamParameters.MIN_DISTANCE_BETWEEN_KEYFRAMES or
                self.rotation_since_keyframe >= SlamParameters.MIN_ROTATION_BETWEEN_KEYFRAMES or
                self.frames_since_keyframe >= SlamParameters.MAX_FRAMES_BETWEEN_KEYFRAMES or
                num_matches < SlamParameters.MIN_KEYFRAME_MATCHES):
                
                # Reset counters
                self.frames_since_keyframe = 0
                self.distance_since_keyframe = 0.0
                self.rotation_since_keyframe = 0.0
                return True
                
        return False

    def _create_keyframe(self, frame: Frame) -> Keyframe:
        """Convert frame to keyframe and reset motion counters."""
        keyframe = Keyframe(frame)
        self.map.add_keyframe(keyframe)
        self.last_keyframe = keyframe
        
        # Reset counters
        self.frames_since_keyframe = 0
        self.distance_since_keyframe = 0.0
        self.rotation_since_keyframe = 0.0
        
        return keyframe
        
    def _update_local_map(self, new_keyframe: Keyframe):
        """Update local map with new keyframe and optimize pose graph."""
        self.local_keyframes.append(new_keyframe)
        
        # Add keyframe to pose graph
        is_first = len(self.pose_graph.keyframes) == 0
        self.pose_graph.add_keyframe(new_keyframe, add_prior=is_first)
        
        # Optimize if we have enough keyframes
        if len(self.pose_graph.keyframes) >= 2:
            print("[Bundle Adjustment] Optimizing poses...")
            if self.pose_graph.optimize():
                print("[Bundle Adjustment] Optimization successful")
                
                # Update map points using optimized poses
                for point_id, point in self.map.map_points.items():
                    # Transform point using ratio of old and new poses
                    ref_kf_id = min(point.observing_keyframes)  # Use first observing keyframe as reference
                    if ref_kf_id in self.map.keyframes:  # <-- Fixed: use self.map.keyframes
                        ref_kf = self.map.keyframes[ref_kf_id]  # <-- Fixed: use self.map.keyframes
                        old_pose = ref_kf.pose.copy()
                        new_pose = ref_kf.pose  # Already updated by pose graph
                        
                        # Transform point from world to local using old pose
                        local_point = np.linalg.inv(old_pose) @ np.append(point.position, 1)
                        # Transform back to world using new pose
                        point.position = (new_pose @ local_point)[:3]
            else:
                print("[Bundle Adjustment] Optimization failed")
        
        # Update local map points
        self.map.update_local_points([kf for kf in self.local_keyframes])

    def get_camera_trajectory(self) -> List[np.ndarray]:
        """Get list of camera poses."""
        return self.map.get_keyframe_poses()
        
    def get_map_points(self) -> Dict[int, np.ndarray]:
        """Get all map points."""
        return self.map.get_all_points()
        
    def __str__(self) -> str:
        status = []
        status.append("SLAM System Status:")
        status.append(f"Initialized: {self.initialized}")
        status.append(f"Device: {self.device}")
        
        if self.initialized:
            status.append("\nTracking:")
            status.append(f"Local keyframes: {len(self.local_keyframes)}/{self.num_local_keyframes}")
            if self.current_frame is not None:
                status.append(f"Current frame: {self.current_frame.id}")
            if self.last_keyframe is not None:
                status.append(f"Last keyframe: {self.last_keyframe.id}")
            status.append(f"Frames since KF: {self.frames_since_keyframe}")
            status.append(f"Distance since KF: {self.distance_since_keyframe:.2f}m")
            status.append(f"Rotation since KF: {self.rotation_since_keyframe:.1f}°")
        
        # Add map status
        status.append("\nMap:")
        status.extend(str(self.map).split("\n")[1:])  # Skip map header
        
        # Add tracker status
        status.append("\nTracker:")
        status.extend(str(self.tracker).split("\n")[1:])  # Skip tracker header
        
        return "\n".join(status)