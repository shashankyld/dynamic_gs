import cv2
import numpy as np
import torch
from thirdparty.LightGlue.lightglue import SuperPoint, LightGlue
from thirdparty.LightGlue.lightglue.utils import rbd
from typing import Optional, Tuple
from core.frame import Frame

class Tracker:
    """Simple frame-to-frame tracker using SuperPoint features and PnP."""
    
    def __init__(self, camera_matrix: np.ndarray, 
                 device: Optional[torch.device] = None,
                 num_features: int = 5000):
        self.camera_matrix = camera_matrix
        self.device = device if device is not None else torch.device("cpu")
        
        # Initialize feature extractor and matcher
        self.extractor = SuperPoint(max_num_keypoints=num_features).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def track_frames(self, frame1: Frame, frame2: Frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Track frame2 relative to frame1 using feature matching and PnP.
        Returns relative pose and inlier indices.
        """
        # Extract features
        with torch.no_grad():
            # Prepare images
            img1 = torch.from_numpy(frame1.image).permute(2,0,1).float()/255.0
            img2 = torch.from_numpy(frame2.image).permute(2,0,1).float()/255.0
            
            # Extract and match features
            feat1 = self.extractor.extract(img1.unsqueeze(0).to(self.device))
            feat2 = self.extractor.extract(img2.unsqueeze(0).to(self.device))
            matches_out = self.matcher({"image0": feat1, "image1": feat2})
            
            # Get keypoints and matches
            feats1, feats2 = [rbd(x) for x in [feat1, feat2]]
            matches = matches_out["matches"][0]

        # Store keypoints in frames
        frame1.keypoints = feats1["keypoints"].cpu().numpy()
        frame2.keypoints = feats2["keypoints"].cpu().numpy()

        # Prepare 3D-2D correspondences for PnP
        obj_points = []  # 3D points from frame1
        img_points = []  # 2D points from frame2
        
        for m in matches:
            idx1, idx2 = int(m[0]), int(m[1])
            kp1 = frame1.keypoints[idx1]
            kp2 = frame2.keypoints[idx2]
            
            # Get depth for keypoint in frame1
            u, v = int(round(kp1[0])), int(round(kp1[1]))
            if u < 0 or v < 0 or v >= frame1.depth.shape[0] or u >= frame1.depth.shape[1]:
                continue
                
            depth = frame1.depth[v, u]
            if depth <= 0:
                continue
                
            # Backproject to 3D
            X = (kp1[0] - self.camera_matrix[0,2]) * depth / self.camera_matrix[0,0]
            Y = (kp1[1] - self.camera_matrix[1,2]) * depth / self.camera_matrix[1,1]
            Z = depth
            
            obj_points.append([X, Y, Z])
            img_points.append(kp2)

        # Check if we have enough points
        if len(obj_points) < 6:
            return None, None

        # Solve PnP
        obj_points = np.array(obj_points, dtype=np.float64)
        img_points = np.array(img_points, dtype=np.float64)
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points,
            self.camera_matrix, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=8.0,
            iterationsCount=100,
            confidence=0.99
        )

        if not success:
            return None, None

        # Convert to transformation matrix
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = tvec.flatten()

        # Update frame poses
        if frame1.pose is None:
            frame1.pose = np.eye(4)
        frame2.pose = frame1.pose @ T

        return T, inliers
