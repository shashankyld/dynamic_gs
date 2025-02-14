import numpy as np

class PoseTracker:
    def __init__(self, use_gt=True):
        self.use_gt = use_gt
        self.poses = {}
        self.skipped_frames = set()
        self.gt_data = None
        
    def set_ground_truth(self, gt_poses, gt_timestamps):
        """Set ground truth data if using GT poses"""
        self.gt_data = {
            'poses': gt_poses,
            'timestamps': gt_timestamps
        }
        
    def get_pose(self, frame_id, timestamp=None):
        """Get pose for a frame, either from GT or tracking"""
        if self.use_gt:
            if self.gt_data is None:
                raise ValueError("Ground truth data not set!")
            return self._get_gt_pose(timestamp)
        else:
            return self._get_tracked_pose(frame_id)
            
    def _get_gt_pose(self, timestamp):
        """Get pose from ground truth data"""
        if timestamp is None:
            return None, False
            
        closest_idx = np.argmin(np.abs(self.gt_data['timestamps'] - timestamp))
        time_diff = abs(self.gt_data['timestamps'][closest_idx] - timestamp)
        
        if time_diff > 0.1:  # More than 100ms difference
            return None, False
            
        return self.gt_data['poses'][closest_idx], True
        
    def _get_tracked_pose(self, frame_id):
        """Get pose from tracking (to be implemented)"""
        # TODO: Implement actual tracking
        return np.eye(4), True
        
    def add_pose(self, frame_id, pose):
        """Store a new pose"""
        self.poses[frame_id] = pose
        
    def get_all_poses(self):
        """Get all stored poses"""
        return self.poses
