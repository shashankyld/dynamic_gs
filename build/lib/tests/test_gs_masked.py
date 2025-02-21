import torch
import numpy as np
import sys
sys.path.append("/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/thirdparty/")
from tqdm import tqdm

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.dataset_bridge import get_camera_info_from_pyslam_dataloader_insteadofgs
from config import Config
from io_utils.dataset import dataset_factory
from io_utils.ground_truth import groundtruth_factory
import cv2
from datetime import datetime
import os
from munch import Munch

def create_simple_mask(height, width, radius_fraction=0.2):
    """Create a circular mask in the center of the image"""
    Y, X = np.ogrid[:height, :width]
    center = (height//2, width//2)
    radius = min(height, width) * radius_fraction
    
    # Create circular mask (1 inside circle, 0 outside)
    mask = ((X - center[1])**2 + (Y - center[0])**2) <= radius**2
    return torch.from_numpy(mask).cuda()

def main():
    # Initialize configuration and dataset
    config = Config()
    dataset = dataset_factory(config)
    ground_truth = groundtruth_factory(config.dataset_settings)
    depth_scale = 1/5000 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load first frame
    img_id = 0
    img = dataset.getImage(img_id)
    depth = dataset.getDepth(img_id) * depth_scale

    # Create mask
    mask = create_simple_mask(img.shape[0], img.shape[1])
    
    # Set up Gaussian model
    sh_degree = 0
    gaussian_model = GaussianModel(sh_degree=sh_degree, config=config)
    gaussian_model.init_lr(6.0)

    # Training settings
    training_params = Munch(config.gs_opt_params)
    pipeline_params = Munch(config.pipeline_params)
    gaussian_model.training_setup(training_params)

    # Get camera info
    camera_info = get_camera_info_from_pyslam_dataloader_insteadofgs(
        dataset, ground_truth, img_id=0, config=config
    )

    # Initialize Gaussian model with masked point cloud
    fused_point_cloud, features, scales, rots, opacities = gaussian_model.create_pcd_from_image_mask(
        camera_info, depthmap=depth, mask=mask
    )
    gaussian_model.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id=0)

    # Set up save directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"masked_{current_time}")
    os.makedirs(save_path, exist_ok=True)

    # Save mask visualization
    mask_vis = mask.cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(os.path.join(save_path, "mask.png"), mask_vis)

    # Training loop
    num_iterations = 1000
    background_color = torch.zeros(3).to(device)

    for iteration in tqdm(range(num_iterations)):
        # Render
        render_output = render(
            camera_info, 
            gaussian_model, 
            pipeline_params, 
            background_color
        )
        
        if render_output is None:
            continue

        rendered_image = render_output["render"]
        viewspace_points = render_output["viewspace_points"]
        visibility_filter = render_output["visibility_filter"]

        # Calculate masked loss
        target_image = torch.from_numpy(img).to(device).float() / 255.0
        target_image = target_image.permute(2, 0, 1)
        loss = gaussian_model.compute_masked_loss(rendered_image, target_image, mask)

        # Optimization step
        gaussian_model.optimizer.zero_grad()
        loss.backward()
        gaussian_model.optimizer.step()

        # Densification and pruning
        gaussian_model.add_densification_stats(viewspace_points, visibility_filter)
        gaussian_model.densify_and_prune(
            max_grad=0.01, 
            min_opacity=0.005, 
            extent=1.0, 
            max_screen_size=10
        )

        # Save checkpoints
        if iteration % 200 == 0:
            rendered_image_cpu = rendered_image.cpu().detach().numpy().transpose(1, 2, 0)
            rendered_image_cpu = (rendered_image_cpu * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f"render_{iteration}.png"), rendered_image_cpu)
            gaussian_model.save_ply(os.path.join(save_path, f"model_{iteration}.ply"))

    # Save final model
    gaussian_model.save_ply(os.path.join(save_path, "final_model.ply"))

if __name__ == "__main__":
    main()
