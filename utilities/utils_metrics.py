import numpy as np

def estimate_error_R_T(T_est, T_gt):
    """Estimate the error in rotation and translation between two poses.
    
    Args:
        T_est (np.ndarray): Estimated pose matrix (4x4).
        T_gt (np.ndarray): Ground truth pose matrix (4x4).
    
    Returns:
        float: Rotation error in degrees.
        float: Translation error in meters.
    """
    # Extract rotation and translation
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    R_gt = T_gt[:3, :3]
    t_gt = T_gt[:3, 3]
    
    # Compute rotation error
    R_err = R_est.T @ R_gt
    trace = np.trace
    trace_ = trace(R_err)
    trace_ = min(3.0, max(-1.0, trace_))
    angle_err = np.arccos((trace_ - 1.0) / 2.0)
    angle_err = np.rad2deg(angle_err)

    # Compute translation error
    t_err = np.linalg.norm(t_gt - t_est)

    return angle_err, t_err
