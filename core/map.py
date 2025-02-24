from typing import Dict, List, Set, Optional, Tuple
import numpy as np
from core.keyframe import Keyframe
import pickle
import os
from datetime import datetime
from scipy.spatial import cKDTree

class MapPoint:
    """3D point in the map with observation info."""
    def __init__(self, point_id: int, position: np.ndarray, descriptor: Optional[np.ndarray] = None):
        self.id = point_id
        self.position = position  # 3D position
        self.observing_keyframes: Set[int] = set()  # Keyframe IDs that see this point
        self.descriptor = descriptor  # Feature descriptor
        self.is_outlier = False
        self.last_observed_time = 0.0
        
    def update_descriptor(self, new_descriptor: np.ndarray):
        """Update descriptor as running average."""
        if self.descriptor is None:
            self.descriptor = new_descriptor
        else:
            self.descriptor = 0.9 * self.descriptor + 0.1 * new_descriptor
            
class Map:
    def __init__(self, local_window_size: int = 7):
        self.keyframes: Dict[int, Keyframe] = {} 
        self.map_points: Dict[int, MapPoint] = {}
        self.local_map_points: Set[int] = set()
        self.local_keyframes: List[int] = []  # Ordered list of recent keyframe IDs
        self.local_window_size = local_window_size
        self.next_point_id = 0
        self.kdtree = None
        self.descriptor_distance_threshold = 0.7  # Descriptor similarity threshold
        self.spatial_distance_threshold = 0.1     # 10cm in 3D space
        self.max_descriptor_candidates = 10       # Maximum number of candidates to check
        
    def find_matching_point(self, position: np.ndarray, descriptor: np.ndarray) -> Optional[int]:
        """Find existing map point that matches given position and descriptor."""
        if len(self.map_points) == 0:
            return None
            
        # Update KD-tree if needed
        if self.kdtree is None:
            positions = np.array([p.position for p in self.map_points.values()])
            self.kdtree = cKDTree(positions)
            
        # Find nearest neighbors within spatial threshold
        distances, indices = self.kdtree.query(position, 
                                             k=min(self.max_descriptor_candidates, len(self.map_points)),
                                             distance_upper_bound=self.spatial_distance_threshold)
        
        if isinstance(distances, float):  # Single result
            distances, indices = [distances], [indices]
            
        # Check descriptor similarity for nearby points
        best_match = None
        best_similarity = float('inf')
        
        for dist, idx in zip(distances, indices):
            if dist == float('inf'):
                continue
                
            point_id = list(self.map_points.keys())[idx]
            map_point = self.map_points[point_id]
            
            if map_point.descriptor is not None and descriptor is not None:
                similarity = np.linalg.norm(map_point.descriptor - descriptor)
                if similarity < self.descriptor_distance_threshold and similarity < best_similarity:
                    best_similarity = similarity
                    best_match = point_id
                    
        return best_match

    def add_keyframe(self, keyframe: Keyframe):
        """Add new keyframe and its points to map, respecting dynamic mask."""
        self.keyframes[keyframe.id] = keyframe
        self.kdtree = None  # Invalidate KD-tree
        
        # Update local keyframes window
        self.local_keyframes.append(keyframe.id)
        if len(self.local_keyframes) > self.local_window_size:
            old_kf_id = self.local_keyframes.pop(0)
            self._remove_from_local_map(old_kf_id)
            
        if keyframe.keypoints is not None and keyframe.depth is not None:
            points_3d = self._backproject_keypoints(keyframe)
            print(f"Processing {len(points_3d)} points from keyframe {keyframe.id}")
            
            new_points_added = 0
            points_matched = 0
            
            for idx, (point_3d, kp) in enumerate(zip(points_3d, keyframe.keypoints)):
                if point_3d is not None:
                    # Check dynamic mask
                    x, y = int(round(kp[0])), int(round(kp[1]))
                    if keyframe.dynamic_mask is not None and not self._is_point_static(x, y, keyframe.dynamic_mask):
                        continue
                    
                    # Transform to world coordinates
                    if keyframe.pose is not None:
                        point_3d = (keyframe.pose @ np.append(point_3d, 1))[:3]
                    
                    descriptor = keyframe.descriptors[idx] if keyframe.descriptors is not None else None
                    
                    # Try to find matching existing point
                    matching_point_id = self.find_matching_point(point_3d, descriptor)
                    
                    if matching_point_id is not None:
                        # Update existing point
                        self.map_points[matching_point_id].observing_keyframes.add(keyframe.id)
                        if descriptor is not None:
                            self.map_points[matching_point_id].update_descriptor(descriptor)
                        keyframe.visible_map_points.add(matching_point_id)
                        points_matched += 1
                    else:
                        # Add new point
                        point_id = self.add_point(point_3d, keyframe, descriptor)
                        keyframe.visible_map_points.add(point_id)
                        new_points_added += 1
            
            print(f"Added {new_points_added} new points, matched {points_matched} existing points")
            
        self.update_local_points()

    def _is_point_static(self, x: int, y: int, mask: np.ndarray) -> bool:
        """Check if a point is in the static region of the mask.
        
        Args:
            x, y: Point coordinates
            mask: Binary mask where True indicates static regions
            
        Returns:
            bool: True if point is in static region
        """
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return mask[y, x]  # True if static, False if dynamic
        return False  # Points outside image bounds considered dynamic

    def _backproject_keypoints(self, keyframe: Keyframe) -> List[Optional[np.ndarray]]:
        """Backproject keypoints to 3D using depth."""
        points_3d = []
        for kp in keyframe.keypoints:
            u, v = int(round(kp[0])), int(round(kp[1]))
            if 0 <= v < keyframe.depth.shape[0] and 0 <= u < keyframe.depth.shape[1]:
                depth = keyframe.depth[v, u]
                if depth > 0:
                    X = (kp[0] - keyframe.cx) * depth / keyframe.fx
                    Y = (kp[1] - keyframe.cy) * depth / keyframe.fy
                    Z = depth
                    points_3d.append(np.array([X, Y, Z]))
                    continue
            points_3d.append(None)
        return points_3d
        
    def _remove_from_local_map(self, keyframe_id: int):
        """Remove keyframe points from local map."""
        if keyframe_id in self.keyframes:
            kf = self.keyframes[keyframe_id]
            self.local_map_points -= kf.visible_map_points
            
    def add_point(self, position: np.ndarray, 
                 observing_keyframe: Keyframe,
                 descriptor: Optional[np.ndarray] = None) -> int:
        """Add new map point and add to local map if from recent keyframe."""
        point_id = self._get_next_point_id()
        point = MapPoint(point_id, position, descriptor)
        point.observing_keyframes.add(observing_keyframe.id)
        point.last_observed_time = observing_keyframe.timestamp
        self.map_points[point_id] = point
        
        # Add to local map if from recent keyframe
        if observing_keyframe.id in self.local_keyframes:
            self.local_map_points.add(point_id)
            
        return point_id
        
    def update_local_points(self, local_keyframes: Optional[List[Keyframe]] = None):
        """Update set of points visible in local keyframes."""
        self.local_map_points.clear()
        
        # If no keyframes provided, use stored local_keyframes
        keyframes_to_use = local_keyframes if local_keyframes is not None else [
            self.keyframes[kf_id] for kf_id in self.local_keyframes 
            if kf_id in self.keyframes
        ]
        
        # Update local map points from all local keyframes
        for kf in keyframes_to_use:
            self.local_map_points.update(kf.visible_map_points)
                
    def merge_point_observations(self, point_id1: int, point_id2: int):
        """Merge two map points that correspond to same 3D point."""
        if point_id1 not in self.map_points or point_id2 not in self.map_points:
            return
            
        point1 = self.map_points[point_id1]
        point2 = self.map_points[point_id2]
        
        # Merge observations
        point1.observing_keyframes.update(point2.observing_keyframes)
        
        # Update keyframes to reference point1
        for kf_id in point2.observing_keyframes:
            if kf_id in self.keyframes:
                kf = self.keyframes[kf_id]
                kf.visible_map_points.remove(point_id2)
                kf.visible_map_points.add(point_id1)
                
        # Update descriptors
        if point2.descriptor is not None:
            point1.update_descriptor(point2.descriptor)
            
        # Remove point2
        del self.map_points[point_id2]
        self.local_map_points.discard(point_id2)
        
    def get_keyframe_poses(self) -> List[np.ndarray]:
        """Get list of all keyframe poses."""
        return [kf.pose for kf in self.keyframes.values()]
        
    def get_all_points(self) -> Dict[int, np.ndarray]:
        """Get all map points as id->position dictionary."""
        return {pid: point.position for pid, point in self.map_points.items()}
        
    def get_local_points(self) -> Dict[int, np.ndarray]:
        """Get points visible in local map."""
        return {pid: self.map_points[pid].position 
                for pid in self.local_map_points 
                if pid in self.map_points}
                
    def _get_next_point_id(self) -> int:
        """Get next available point ID."""
        point_id = self.next_point_id
        self.next_point_id += 1
        return point_id

    def __str__(self) -> str:
        """Enhanced string representation with local map info."""
        status = []
        status.append("Map Status:")
        status.append(f"Keyframes: {len(self.keyframes)} (Local: {len(self.local_keyframes)})")
        status.append(f"Map points: {len(self.map_points)} (Local: {len(self.local_map_points)})")
        
        if self.map_points:
            points = np.array([p.position for p in self.map_points.values()])
            min_xyz = points.min(axis=0)
            max_xyz = points.max(axis=0)
            status.append("\nMap bounds:")
            status.append(f"X: [{min_xyz[0]:.1f}, {max_xyz[0]:.1f}]")
            status.append(f"Y: [{min_xyz[1]:.1f}, {max_xyz[1]:.1f}]")
            status.append(f"Z: [{min_xyz[2]:.1f}, {max_xyz[2]:.1f}]")
            
        return "\n".join(status)
    
    def save(self, save_dir: str = "logs/slam_map"):
        """Save map object to file."""
        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, timestamp)
        os.makedirs(save_path, exist_ok=True)
        
        # Prepare data for saving
        map_data = {
            'keyframes': self.keyframes,
            'map_points': self.map_points,
            'local_map_points': self.local_map_points,
            'local_keyframes': self.local_keyframes,
            'local_window_size': self.local_window_size,
            'next_point_id': self.next_point_id
        }
        
        # Save map data
        filepath = os.path.join(save_path, "map.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(map_data, f)
        
        print(f"Map saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str):
        """Load map object from file."""
        with open(filepath, 'rb') as f:
            map_data = pickle.load(f)
            
        map_obj = cls(local_window_size=map_data['local_window_size'])
        map_obj.keyframes = map_data['keyframes']
        map_obj.map_points = map_data['map_points']
        map_obj.local_map_points = map_data['local_map_points']
        map_obj.local_keyframes = map_data['local_keyframes']
        map_obj.next_point_id = map_data['next_point_id']
        
        print(f"Map loaded from {filepath}")
        return map_obj