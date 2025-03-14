class SlamParameters:   

    # Number of desired keypoints per frame 
    kNumFeatures=5000
    
    # Visualization 
    kShowDebugImages = True

    # Dynamic SLAM
    kNumFramesAway = 5
    kDynamicEdgeThreshold = 0.05
    kDynamicEdgeThresholdVar = 0.05
    kMaxEdgeLength = 0.5
    kDynamicProcessingMethod = 'batch'  ## 'batch' or 'k_frame_away' 
    kDynamicBatchSize = 3  # 2
    kDynamicStride = 2      # 4

    # STARTING AND ENDING FRAME
    kStartingFrameIdx = 250
    kEndingFrameIdx = 330
    # Keyframe creation parameters
    MIN_DISTANCE_BETWEEN_KEYFRAMES = 1  # meters
    MIN_ROTATION_BETWEEN_KEYFRAMES = 20.0  # degrees
    MAX_FRAMES_BETWEEN_KEYFRAMES = 30      # frames
    MIN_INLIER_THRESHOLD = 30             # minimum inliers for good tracking
    MIN_KEYFRAME_MATCHES = 100            # minimum matches before forcing new keyframe

    # Local mapping parameters
    NUM_LOCAL_KEYFRAMES = 7               # size of local keyframe window
    NUM_FEATURES = 5000                   # number of features to extract