import numpy as np
from typing import Dict, List, Set, Optional
from .frame import Frame

class MapPoint:
    def __init__(self, id: int, position: np.ndarray, descriptor: np.ndarray):
        self.id = id
        self.position = position  # 3D position
        self.descriptor = descriptor
        self.observations: Dict[int, int] = {}  # frame_id -> keypoint_idx
        self.is_dynamic = False
        self.reference_keyframe_id = -1
        
    def add_observation(self, frame_id: int, keypoint_idx: int):
        self.observations[frame_id] = keypoint_idx
        
    def remove_observation(self, frame_id: int):
        if frame_id in self.observations:
            del self.observations[frame_id]

class Map:
    def __init__(self):
        self.points: Dict[int, MapPoint] = {}
        self.next_point_id = 0
        self.keyframes: Dict[int, Frame] = {}
        
    def add_point(self, position: np.ndarray, descriptor: np.ndarray,
                 frame_id: int, keypoint_idx: int) -> int:
        """Add a new map point and return its ID"""
        point_id = self.next_point_id
        self.next_point_id += 1
        
        point = MapPoint(point_id, position, descriptor)
        point.add_observation(frame_id, keypoint_idx)
        point.reference_keyframe_id = frame_id
        self.points[point_id] = point
        
        return point_id
        
    def remove_point(self, point_id: int):
        """Remove a map point"""
        if point_id in self.points:
            del self.points[point_id]
            
    def get_points_visible_in_frame(self, frame: Frame) -> Set[int]:
        """Get IDs of map points potentially visible in frame"""
        return frame.get_visible_map_points(
            {pid: p.position for pid, p in self.points.items()})
    
    def update_point_position(self, point_id: int, position: np.ndarray):
        """Update position of a map point"""
        if point_id in self.points:
            self.points[point_id].position = position
            
    def mark_point_dynamic(self, point_id: int):
        """Mark a map point as dynamic"""
        if point_id in self.points:
            self.points[point_id].is_dynamic = True
            
    def add_keyframe(self, frame: Frame):
        """Add a keyframe to the map"""
        self.keyframes[frame.id] = frame
        
    def get_local_map_points(self, frame_id: int, radius: int = 3) -> Set[int]:
        """Get map points visible in nearby keyframes"""
        local_points = set()
        if frame_id in self.keyframes:
            # Get connected keyframes
            connected_kfs = set()
            for kf_id, weight in self.keyframes[frame_id].connected_keyframes.items():
                if weight > 0:
                    connected_kfs.add(kf_id)
                    
            # Get points observed by connected keyframes
            for kf_id in connected_kfs:
                if kf_id in self.keyframes:
                    local_points.update(self.keyframes[kf_id].visible_map_points)
                    
        return local_points
