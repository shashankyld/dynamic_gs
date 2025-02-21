import cv2
import numpy as np
from typing import Tuple, Optional
from core.frame import Frame

class Tracker:
    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None,
                 pnp_iterations: int = 100, reprojection_error: float = 8.0, confidence: float = 0.99):
        """
        Initialize Tracker.
        
        Args:
            camera_matrix: Intrinsic camera matrix.
            dist_coeffs: Distortion coefficients, default to zeros.
            pnp_iterations: Maximum iterations for RANSAC.
            reprojection_error: Reprojection error threshold.
            confidence: Confidence value for RANSAC.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))
        self.pnp_iterations = pnp_iterations
        self.reprojection_error = reprojection_error
        self.confidence = confidence

    def track_frame(self, last_frame: Frame, current_frame: Frame, mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Track current frame with the last frame using PnP (ignoring masked dynamic areas).

        Args:
            last_frame: The previous keyframe (with valid depth and features).
            current_frame: The current frame (with detected 2D keypoints).
            mask: A binary mask image (0=static, nonzero=dynamic). Only keypoints landing on zero will be used.
            
        Returns:
            relative_pose: 4x4 relative transformation matrix (from last to current).
            inliers: Indices of inlier correspondences.
        """
        # Ensure keypoints exist
        if last_frame.keypoints is None or current_frame.keypoints is None:
            return None, None

        object_points = []
        image_points = []
        
        # For simplicity, assume correspondence by index (this can be replaced with descriptor matching)
        num_points = min(len(last_frame.keypoints), len(current_frame.keypoints))
        for i in range(num_points):
            # Filter current frame keypoint using mask (assume mask is same resolution as image)
            kp = current_frame.keypoints[i]
            x, y = int(round(kp[0])), int(round(kp[1]))
            if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
                continue
            if mask[y, x] != 0:  # non-zero in mask indicates dynamic region; skip it
                continue
            # Compute corresponding 3D point from last frame depth using pinhole model
            d = last_frame.depth[int(round(last_frame.keypoints[i][1])), int(round(last_frame.keypoints[i][0]))]
            cx, cy = last_frame.cx, last_frame.cy
            fx, fy = last_frame.fx, last_frame.fy
            X = (last_frame.keypoints[i][0] - cx) * d / fx
            Y = (last_frame.keypoints[i][1] - cy) * d / fy
            Z = d
            object_points.append([X, Y, Z])
            image_points.append([kp[0], kp[1]])
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        # Need a sufficient number of correspondences to solve PnP (at least 6 recommended)
        if object_points.shape[0] < 6:
            return None, None

        # Run PnP (using RANSAC) for 3D-2D correspondence
        success, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points,
                                                             self.camera_matrix, self.dist_coeffs,
                                                             flags=cv2.SOLVEPNP_ITERATIVE,
                                                             reprojectionError=self.reprojection_error,
                                                             iterationsCount=self.pnp_iterations,
                                                             confidence=self.confidence)
        if not success or inliers is None:
            return None, None

        R, _ = cv2.Rodrigues(rvec)
        relative_pose = np.eye(4)
        relative_pose[:3, :3] = R
        relative_pose[:3, 3] = tvec.flatten()
        return relative_pose, inliers
