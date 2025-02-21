import numpy as np
import gtsam

def test_gtsam():
    try:
        print("Testing GTSAM installation...")
        
        # Create a simple pose graph
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()
        
        # Add first pose with prior
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        
        x1 = gtsam.symbol('x', 1)
        pose1 = gtsam.Pose3()  # Identity pose
        graph.add(gtsam.PriorFactorPose3(x1, pose1, prior_noise))
        initial.insert(x1, pose1)
        
        # Add second pose with relative constraint
        x2 = gtsam.symbol('x', 2)
        pose2 = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1, 0, 0))  # 1m in x direction
        between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        graph.add(gtsam.BetweenFactorPose3(x1, x2, pose2, between_noise))
        initial.insert(x2, pose2)
        
        # Optimize
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
        result = optimizer.optimize()
        
        # Check results
        if result.exists(x1) and result.exists(x2):
            final_pose2 = result.atPose3(x2)
            print(f"First pose position: {result.atPose3(x1).translation()}")
            print(f"Second pose position: {final_pose2.translation()}")
            print("GTSAM test successful!")
        else:
            print("Error: Optimization failed - poses not found in result")
            
    except Exception as e:
        print(f"GTSAM test failed with error: {str(e)}")

if __name__ == "__main__":
    test_gtsam()
