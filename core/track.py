""" 
When modified, test by running: 
1. tests/test_track_core.py 
2. tests/test_track_dynamic_core.py
"""

import cv2
import numpy as np
import torch
from thirdparty.LightGlue.lightglue import SuperPoint, LightGlue
from thirdparty.LightGlue.lightglue.utils import rbd
from typing import Optional, Tuple
from core.frame import Frame

def convert_image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert image to normalized torch tensor with correct channel order."""
    if img.ndim == 2:  # Grayscale
        # Add channel dimension and repeat to make 3 channels
        img_tensor = torch.from_numpy(img).float()[None] / 255.0
        img_tensor = img_tensor.repeat(3, 1, 1)  # Convert to 3 channels
    else:  # RGB
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img_tensor

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

    def extract_features(self, frame: Frame) -> bool:
        """Extract features from frame if not already present."""
        if frame.keypoints is not None:
            return True
            
        if frame.image is None:
            return False
            
        with torch.no_grad():
            img = convert_image_to_tensor(frame.image).unsqueeze(0).to(self.device)
            feat = self.extractor.extract(img)
            feats = rbd(feat)
            
            frame.keypoints = feats["keypoints"].cpu().numpy()
            frame.descriptors = feats["descriptors"].cpu().numpy()
            
        return True

    def track_frames(self, frame1: Frame, frame2: Frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Track frame2 relative to frame1 using feature matching and PnP."""
        # Check for required data
        if (not self.extract_features(frame1) or not self.extract_features(frame2) or 
            frame1.depth is None or frame2.depth is None):
            print("[Tracker] Missing features or depth data")
            return None, None

        with torch.no_grad():
            # Convert images to tensors
            img1 = convert_image_to_tensor(frame1.image).unsqueeze(0).to(self.device)
            img2 = convert_image_to_tensor(frame2.image).unsqueeze(0).to(self.device)
            
            # Extract and match features
            feat1 = self.extractor.extract(img1)
            feat2 = self.extractor.extract(img2)
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
        valid_matches = []
        
        for m in matches:
            idx1, idx2 = int(m[0]), int(m[1])
            kp1 = frame1.keypoints[idx1]
            kp2 = frame2.keypoints[idx2]
            
            # Get depth for keypoint in frame1
            u, v = int(round(kp1[0])), int(round(kp1[1]))
            
            # Add boundary and depth checks
            if not (0 <= v < frame1.depth.shape[0] and 0 <= u < frame1.depth.shape[1]):
                continue
                
            depth = frame1.depth[v, u]
            if depth <= 0 or not np.isfinite(depth):
                continue
                
            # Backproject to 3D
            X = (kp1[0] - self.camera_matrix[0,2]) * depth / self.camera_matrix[0,0]
            Y = (kp1[1] - self.camera_matrix[1,2]) * depth / self.camera_matrix[1,1]
            Z = depth
            
            # Skip points with invalid 3D coordinates
            if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
                continue
                
            obj_points.append([X, Y, Z])
            img_points.append(kp2)
            valid_matches.append([idx1, idx2])

        # Check if we have enough points
        if len(obj_points) < 6:
            print(f"[Tracker] Not enough valid matches: {len(obj_points)} < 6")
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

    def track_frames_with_mask(self, frame1: Frame, frame2: Frame, 
                             mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Track frame2 relative to frame1 using only features within the mask."""
        # Check for required data
        if (not self.extract_features(frame1) or not self.extract_features(frame2) or 
            frame1.depth is None or frame2.depth is None):
            print("[Tracker] Missing features or depth data")
            return None, None

        # Ensure both frames have features
        if not self.extract_features(frame1) or not self.extract_features(frame2):
            return None, None

        with torch.no_grad():
            # Convert images to tensors
            img1 = convert_image_to_tensor(frame1.image).unsqueeze(0).to(self.device)
            img2 = convert_image_to_tensor(frame2.image).unsqueeze(0).to(self.device)
            
            feat1 = self.extractor.extract(img1)
            feat2 = self.extractor.extract(img2)
            
            # Get keypoints and descriptors
            feats1, feats2 = [rbd(x) for x in [feat1, feat2]]
            kpts1 = feats1["keypoints"].cpu().numpy()
            kpts2 = feats2["keypoints"].cpu().numpy()
            desc1 = feats1["descriptors"].cpu().numpy()  # Move descriptors to CPU
            
            # Filter keypoints and descriptors using mask
            valid_kpts1 = []
            valid_indices1 = []
            valid_desc1 = []
            
            for i, kp in enumerate(kpts1):
                x, y = int(round(kp[0])), int(round(kp[1]))
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    valid_kpts1.append(kp)
                    valid_indices1.append(i)
                    valid_desc1.append(desc1[i])
                    
            if len(valid_kpts1) < 6:
                return None, None
                
            # Create masked features dictionary
            valid_kpts1 = np.array(valid_kpts1)
            valid_desc1 = np.stack(valid_desc1)  # Now works because desc1 is on CPU
            
            # Convert back to torch tensors on device
            masked_feat1 = {
                "keypoints": torch.from_numpy(valid_kpts1).unsqueeze(0).to(self.device),
                "descriptors": torch.from_numpy(valid_desc1).unsqueeze(0).to(self.device),
                "image_size": feat1["image_size"],
            }
            
            # Match features
            matches_out = self.matcher({
                "image0": masked_feat1,
                "image1": feat2
            })
            matches = matches_out["matches"][0]

        # Store keypoints and continue with PnP (rest remains the same)
        frame1.keypoints = kpts1
        frame2.keypoints = kpts2
        frame2._dynamic_mask = mask

        # Prepare 3D-2D correspondences for PnP using only valid matches
        obj_points = []
        img_points = []
        valid_matches = []
        
        for m in matches:
            idx1 = valid_indices1[int(m[0])]  # Map back to original indices
            idx2 = int(m[1])
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
            valid_matches.append([idx1, idx2])

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

        # Return transformation and inliers
        return T, inliers

    def __str__(self) -> str:
        """Return human-readable tracker status."""
        status = []
        status.append("Tracker Configuration:")
        status.append(f"Device: {self.device}")
        status.append(f"Number of features: {self.extractor.conf.max_num_keypoints}")  # Fixed attribute name
        status.append(f"Feature extractor: SuperPoint")
        status.append(f"Matcher: LightGlue")
        
        # Camera parameters
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]
        status.append(f"\nCamera Parameters:")
        status.append(f"fx={fx:.1f}, fy={fy:.1f}")
        status.append(f"cx={cx:.1f}, cy={cy:.1f}")
        
        return "\n".join(status)
