import numpy as np

class PoseTracker:
    def __init__(self, use_gt=True, max_time_diff=0.1):
        self.poses = {}
        self.use_gt = use_gt
        self.max_time_diff = max_time_diff
        self.gt_poses = None
        self.gt_timestamps = None

    def initialize_gt(self, gt_poses, gt_timestamps):
        if self.use_gt:
            self.gt_poses = gt_poses
            self.gt_timestamps = gt_timestamps
            return True
        return False

    def process_frame(self, frame_id, timestamp=None, point_cloud=None):
        """Process a new frame and return (success, pose)"""
        if self.use_gt:
            return self._get_gt_pose(timestamp)
        else:
            return self._track_pose(frame_id, point_cloud)

    def _get_gt_pose(self, timestamp):
        if timestamp is None or self.gt_timestamps is None:
            return False, None

        closest_idx = np.argmin(np.abs(self.gt_timestamps - timestamp))
        time_diff = abs(self.gt_timestamps[closest_idx] - timestamp)
        
        if time_diff > self.max_time_diff:
            return False, None

        pose = self.gt_poses[closest_idx]
        self.poses[closest_idx] = pose
        return True, pose

    def _track_pose(self, frame_id, point_cloud):
        """To be implemented for custom tracking"""
        return False, None

    def get_all_poses(self):
        return self.poses
