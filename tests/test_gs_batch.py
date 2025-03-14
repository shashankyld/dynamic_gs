import torch
import numpy as np
import open3d as o3d
import sys
sys.path.append("/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/")
sys.path.append("/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/thirdparty/")
from tqdm import tqdm
import os
import cv2
from datetime import datetime

# SLAM related imports
from config import Config
from io_utils.dataset import dataset_factory, SensorType
from io_utils.ground_truth import groundtruth_factory
from utilities.utils_depth import depth2pointcloud, depth2pcd
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint
from utilities.dataset_bridge import get_frame_from_pyslam_dataloader, get_camera_info_from_pyslam_dataloader_insteadofgs
from core.slam_system import SLAMSystem as SLAM
from utilities.utils_metrics import estimate_error_R_T
from core.trajectory_aux import Trajectory
from core.pc_map_aux import PC_Map

# Gaussian Splatting related imports
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.general_utils import build_rotation
from thirdparty.gaussian_splatting.utils.camera_utils import Camera
from thirdparty.gaussian_splatting.utils.graphics_utils import getWorld2View2, BasicPointCloud
from thirdparty.gaussian_splatting.utils.sh_utils import RGB2SH
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from munch import Munch

if __name__ == "__main__":
    print("Starting SLAM and Gaussian Splatting batch mode...")
    
    # PART 1: SLAM SETUP AND MAP BUILDING
    # ===================================
    
    # Set up dataset parameters
    config = Config()
    dataset = dataset_factory(config) 
    depthmapfactor = config.cam_settings["DepthMapFactor"]
    depth_scale = 1/depthmapfactor

    # Camera matrix (intrinsics)
    camera_matrix = np.array([
        [config.cam_settings["Camera.fx"], 0, config.cam_settings["Camera.cx"]],
        [0, config.cam_settings["Camera.fy"], config.cam_settings["Camera.cy"]],
        [0, 0, 1]
    ], dtype=np.float64)

    # GT DATA
    groundtruth = groundtruth_factory(config.dataset_settings)
    
    # PIPELINE DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # SETTING UP SLAM PARAMETERS
    num_features = config.num_features_to_extract if config.num_features_to_extract > 0 else 5000
    kNumFramesAway = config.NumFramesAway
    kNumLocalKFs = config.NUM_LOCAL_KEYFRAMES

    # Feature extraction and matching setup
    slam = SLAM(camera_matrix, num_features, num_local_keyframes=kNumLocalKFs, device=device)

    # Set config to SLAM object
    slam.dataset = dataset  
    slam.groundtruth = groundtruth
    slam.config = config

    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Initialize processing variables
    starting_img_id = config.start_frame_id
    ending_img_id = config.end_frame_id
    img_id = starting_img_id
    global_traj = Trajectory()
    local_traj = Trajectory()
    
    print(f"Processing frames {starting_img_id} to {ending_img_id} to build SLAM map...")
    
    # Process frames to build map
    while img_id <= ending_img_id: 
        if dataset.isOk(): 
            img = dataset.getImage(img_id)
            depth_img = dataset.getDepth(img_id) * depth_scale
            
            if img is not None:
                timestamp = dataset.getTimestamp()
                timestamp, global_traj.trajectory[img_id] = groundtruth.getTimestampPoseMatrix(img_id, set_id_to_eye4=starting_img_id)
                
                print(f"Processing frame: {img_id}")
                
                if img_id == starting_img_id: 
                    # Initialize SLAM with first frame
                    curr_frame = get_frame_from_pyslam_dataloader(dataset, groundtruth, img_id, config)
                    slam.initialize(curr_frame)
                    local_traj.trajectory[img_id] = curr_frame.pose
                    prev_frame = curr_frame
                else:
                    # Process subsequent frames
                    curr_frame = get_frame_from_pyslam_dataloader(dataset, groundtruth, img_id, config)
                    dynamic_mask = None
                    curr_frame._dynamic_mask = dynamic_mask
                    
                    # Track features
                    tracking_result = slam.track_frame(curr_frame)
                    local_traj.trajectory[img_id] = curr_frame.pose
                    
                    # Calculate and print errors
                    tracked_odometry = np.linalg.inv(prev_frame.pose) @ curr_frame.pose
                    gt_odometry = np.linalg.inv(global_traj.trajectory[prev_frame.id]) @ global_traj.trajectory[curr_frame.id]
                    print("Relative Odometry Error deg/m: ", estimate_error_R_T(tracked_odometry, gt_odometry))
                    print("Relative Pose Error deg/m: ", estimate_error_R_T(curr_frame.pose, global_traj.trajectory[curr_frame.id]))
                    
                    prev_frame = curr_frame
        
        img_id += 1

    print(f"SLAM map built with {len(slam.map.keyframes)} keyframes")

    # PART 2: GAUSSIAN SPLATTING WITH BATCH OPTIMIZATION
    # =================================================
    
    print("Initializing Gaussian Splatting with map keyframes...")
    
    # Setup Gaussian Model
    sh_degree = 0
    render_width = dataset.width
    render_height = dataset.height
    background_color = torch.zeros(3).to(device)
    
    # Create Gaussian Model
    gaussian_model = GaussianModel(sh_degree=sh_degree, config=config)
    gaussian_model.init_lr(6.0)
    
    # Setup training parameters
    training_params = config.gs_opt_params
    pipeline_params = config.pipeline_params
    training_args = Munch(training_params)
    pipeline_args = Munch(pipeline_params)
    gaussian_model.training_setup(training_args)
    
    # Create output directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), "logs", f"batch_{current_time}")
    mkdir_p(save_path)
    
    # Get camera info for all keyframes
    keyframes = slam.map.keyframes
    keyframe_ids = list(keyframes.keys())
    print(f"Found {len(keyframe_ids)} keyframes to use for Gaussian Splatting")
    
    # Initialize camera info and data storage for keyframes
    camera_infos = []
    target_images = []
    target_depths = []
    depth_masks = []
    
    # Process all keyframes to initialize the Gaussian model
    print("Adding point clouds from all keyframes to initialize Gaussian model...")
    
    for kf_id in keyframe_ids:
        kf = keyframes[kf_id]
        
        # Get image and depth for keyframe
        img = dataset.getImage(kf_id)
        depth = dataset.getDepth(kf_id) * depth_scale
        
        # Get camera info for keyframe
        camera_info = get_camera_info_from_pyslam_dataloader_insteadofgs(dataset, groundtruth, kf_id, config)
        
        # The camera coordinates in SLAM are typically world-to-camera transform
        # but Gaussian Splatting might expect camera-to-world transform.
        # Using the inverse of the pose for both initialization and rendering
        pose_inverse = np.linalg.inv(kf.pose)
        
        # Convert numpy arrays to torch tensors for camera pose (use inverse pose)
        camera_info.R = torch.from_numpy(pose_inverse[:3, :3]).float().to(device)
        camera_info.T = torch.from_numpy(pose_inverse[:3, 3]).float().to(device)
        
        # Debug info
        print(f"KF {kf_id} original pose:")
        print(kf.pose)
        print(f"KF {kf_id} inverse pose (using for both initialization and rendering):")
        print(pose_inverse)
        
        # Initialize or extend the Gaussian model with this keyframe's point cloud
        print(f"Adding point cloud from keyframe {kf_id}")
        fused_point_cloud, features, scales, rots, opacities = gaussian_model.create_pcd_from_image(camera_info, depthmap=depth)
        gaussian_model.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id=kf_id)
        
        # Store camera info with inverted pose for rendering during optimization
        camera_infos.append(camera_info)  # Already has the inverted pose
        
        # Convert image to tensor
        target_image = torch.from_numpy(img).to(device).float() / 255.0
        target_image = target_image.permute(2, 0, 1)
        target_images.append(target_image)
        
        # Convert depth to tensor and create mask
        target_depth = torch.from_numpy(depth).to(device).float()
        target_depth = target_depth.unsqueeze(0)
        depth_mask = (target_depth > 0).float()
        
        target_depths.append(target_depth)
        depth_masks.append(depth_mask)
    
    print(f"Loaded {len(camera_infos)} keyframes for batch optimization")
    print("Starting Gaussian Splatting optimization...")
    # SAVE THE INITIAL MODEL 
    gaussian_model.save_ply(os.path.join(save_path, f"model_iter0.ply"))
    
    # Set number of epochs and calculate total iterations
    num_epochs = 20  # Adjust as needed
    num_iterations = num_epochs * len(camera_infos)
    
    # Training loop for random keyframe selection optimization
    for iteration in tqdm(range(num_iterations)):
        # Reset gradients for this iteration
        gaussian_model.optimizer.zero_grad()
        
        # Randomly select a keyframe for this iteration
        idx = np.random.randint(0, len(camera_infos))
        camera_info = camera_infos[idx]
        target_image = target_images[idx]
        target_depth = target_depths[idx]
        depth_mask = depth_masks[idx]
        
        # Current epoch tracking
        current_epoch = iteration // len(camera_infos)
        keyframe_in_epoch = iteration % len(camera_infos)
        
        if keyframe_in_epoch == 0:
            print(f"\nStarting epoch {current_epoch + 1}/{num_epochs}")
        
        # Render from this keyframe's perspective
        render_output = render(camera_info, gaussian_model, pipeline_args, background_color)
        
        if render_output is None:
            continue  # Skip if no gaussians to render
        
        rendered_image = render_output["render"]
        viewspace_points = render_output["viewspace_points"]
        visibility_filter = render_output["visibility_filter"]
        rendered_depth = render_output["depth"]
        
        # Calculate image loss
        image_loss = torch.nn.functional.l1_loss(rendered_image, target_image)
        
        # Calculate depth loss
        masked_target_depth = target_depth * depth_mask
        masked_rendered_depth = rendered_depth.squeeze(0) * depth_mask
        depth_loss = torch.nn.functional.l1_loss(masked_rendered_depth, masked_target_depth)
        
        # Total loss
        loss = image_loss + depth_loss
        
        # Backpropagate and update
        loss.backward()
        gaussian_model.optimizer.step()
        
        # Add densification stats
        try:
            gaussian_model.add_densification_stats(viewspace_points, visibility_filter)
        except Exception as e:
            print(f"Warning: Could not add densification stats at iteration {iteration}: {e}")
        
        # Densification and pruning
        if iteration % 100 == 0:
            try:
                gaussian_model.densify_and_prune(max_grad=0.01, min_opacity=0.005, extent=1.0, max_screen_size=10)
            except Exception as e:
                print(f"Warning: Could not perform densification and pruning at iteration {iteration}: {e}")
        
        # Print losses
        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, Keyframe: {keyframe_ids[idx]}, Loss: {loss.item()}, Image Loss: {image_loss.item()}, Depth Loss: {depth_loss.item()}")
        
        # Save checkpoints and renders periodically
        if iteration % 500 == 0:
            # Render and save from a few keyframe perspectives for visualization
            sample_indices = [0, len(camera_infos)//2, -1]
            for i, view_idx in enumerate(sample_indices):
                render_output = render(camera_infos[view_idx], gaussian_model, pipeline_args, background_color)
                if render_output is not None:
                    rendered_image = render_output["render"]
                    rendered_image_cpu = rendered_image.cpu().detach().numpy().transpose(1, 2, 0)
                    rendered_image_cpu = (rendered_image_cpu * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_path, f"render_iter{iteration}_view{i}.png"), rendered_image_cpu)
            
            # Save model checkpoint
            gaussian_model.save_ply(os.path.join(save_path, f"model_iter{iteration}.ply"))
    
    # Save final model
    gaussian_model.save_ply(os.path.join(save_path, "final_model.ply"))
    print(f"Finished! Final model saved to {os.path.join(save_path, 'final_model.ply')}")
