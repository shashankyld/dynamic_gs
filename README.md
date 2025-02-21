1. First simplify neighbourhood matches by removing current delaunay for every frame. track directly with the delaunay from the ref frame.
1.1. Think about a good way to use batch of frames to come up with prompts
2. Use the the strategy to also track # common features that can help decide keyframe spawn decision
3. Implement tracking for RGBD - frame to frame, frame to sparse map, image to GS map, motion model (for failure cases)
4. Setup factor graph slam using gtsam or g20, checkout pypose.
5. Use SAM2 for two purpose - propose dynamic mask, help in improving prompts from the delaunay thing by consistency
6. Classes - Unified Dataloader for SLAM and GS, Frame, KeyFrames, Feature Det/Match, GS MAP, Track, SLAM, TimingUtils
7. Importantly - increase focus on speed metric from the start - track every methods time
8. Sparse Map - List of KeyFrames, Keyframe - [Frame from the DataLoader - But fully populated with Features, Tracked Pose, Results of DynamicFeature Detection from Delaunay based methos, Dynamic Object Mask from SAM2...]
9. Dense Map - List of Gaussians - Each Gaussian is associated with a KeyFrame so upon loopclosure, The gaussians can be deformed as well. Impliment proper - spawn, prune, densify methods for GS MAP. Use dynamic mask to control GS map optimization
10. Focus on building metric from step1. 
11. Testing Framework
    - Unit tests for core components
    - Integration tests for SLAM pipeline
    - Benchmark suite for performance metrics
12. Visualization Components
    - Real-time trajectory visualization
    - Map quality assessment tools
    - Dynamic object detection visualization
13. Robustness
    - Failure recovery mechanisms
    - Edge case handling
    - Tracking quality metrics
14. Documentation
    - API documentation
    - Usage examples
    - Performance benchmarks
15. First impliment individual components, but for the overall slam - exploit parallelism
## Complimentary
1. Make custom dataset that can demonstrate open-set dynamic object removal quality - Bonn IPB doesnt have a lot of dynamic objects in the scene.

## Implementation Timeline
1. Core Components
   - [DONE] Feature detection/matching with Delaunay
   - [REIMPLIMENT CURRENT VERSION] frame-to-frame tracking
   - [ ] Initial timing infrastructure

2. SLAM Backend
   - [ ] Factor graph implementation
   - [ ] Keyframe management
   - [ ] Loop closure detection

3. Dynamic Objects
   - [DONE - NOT INTEGRATED] SAM2 integration
   - [ ] Dynamic mask generation
   - [ ] Tracking refinement

4. Gaussian Splatting
   - [DONE] GS map initialization
   - [DONE] Map optimization
   - [DONE] Dynamic object handling

## Performance Targets
- Feature matching: < Xms per frame
- Tracking: < Xms per frame
- Map update: < Xms per keyframe
- Dynamic object detection: < Xms per frame

## Parallel Processing Strategy
1. Main Thread
   - Feature detection/matching
   - Pose estimation
   - Core SLAM operations

2. Worker Threads
   - SAM2 processing
   - Gaussian Splatting optimization
   - Loop closure detection
   - Map maintenance
