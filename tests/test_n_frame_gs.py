import torch
import numpy as np
import sys
import open3d as o3d
sys.path.append("/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/")
from datetime import datetime
import os
from tqdm import tqdm
import cv2
from munch import Munch

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.dataset_bridge import get_camera_info_from_pyslam_dataloader_insteadofgs

from config import Config
from io_utils.dataset import dataset_factory
from io_utils.ground_truth import groundtruth_factory

class MultiFrameTrainer:
    def __init__(self, config, num_frames=5, frame_skip=10, visualize_init=False):
        self.config = config
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visualize_init = visualize_init
        
        # Initialize dataset
        self.dataset = dataset_factory(config)
        self.ground_truth = groundtruth_factory(config.dataset_settings)
        self.depth_scale = 1/5000
        
        # Training parameters
        self.sh_degree = 0
        self.num_iterations = 2000
        self.background_color = torch.zeros(3).to(self.device)
        
        # Initialize save directory
        self.current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.save_path = os.path.join(os.getcwd(), f"multiframe_{self.current_time}")
        os.makedirs(self.save_path, exist_ok=True)
        
        # Training settings
        self.training_params = Munch(config.gs_opt_params)
        self.pipeline_params = Munch(config.pipeline_params)

    def load_frames(self):
        """Load N frames that are frame_skip apart"""
        self.frames_data = []
        for i in range(self.num_frames):
            img_id = i * self.frame_skip
            img = self.dataset.getImage(img_id)
            depth = self.dataset.getDepth(img_id) * self.depth_scale
            camera_info = get_camera_info_from_pyslam_dataloader_insteadofgs(
                self.dataset, self.ground_truth, img_id, self.config
            )
            
            self.frames_data.append({
                'img_id': img_id,
                'img': torch.from_numpy(img).to(self.device).float() / 255.0,
                'depth': depth,
                'camera_info': camera_info
            })
            print(f"Loaded frame {img_id}")

    def visualize_point_cloud(self, points, colors=None):
        """Visualize point cloud using Open3D"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

    def initialize_gaussians(self):
        """Initialize Gaussian model with points from all frames"""
        self.gaussian_model = GaussianModel(sh_degree=self.sh_degree, config=self.config)
        self.gaussian_model.init_lr(6.0)
        self.gaussian_model.training_setup(self.training_params)
        
        # Collect points from all frames
        all_points = []
        all_features = []
        all_scales = []
        all_rots = []
        all_opacities = []
        
        for frame_data in self.frames_data:
            points, features, scales, rots, opacities = self.gaussian_model.create_pcd_from_image(
                frame_data['camera_info'], 
                depthmap=frame_data['depth']
            )
            all_points.append(points)
            all_features.append(features)
            all_scales.append(scales)
            all_rots.append(rots)
            all_opacities.append(opacities)
        
        # Concatenate all points
        fused_points = torch.cat(all_points, dim=0)
        fused_features = torch.cat(all_features, dim=0)
        fused_scales = torch.cat(all_scales, dim=0)
        fused_rots = torch.cat(all_rots, dim=0)
        fused_opacities = torch.cat(all_opacities, dim=0)
        
        # Visualize if flag is set
        if self.visualize_init:
            # Extract RGB colors from features (first 3 components of SH coefficients)
            colors = fused_features[:, :3, 0]  # Shape: (N, 3)
            self.visualize_point_cloud(fused_points, colors)
        
        # Initialize the model with all points
        self.gaussian_model.extend_from_pcd(
            fused_points, fused_features, fused_scales, fused_rots, fused_opacities, kf_id=0
        )

    def train(self):
        """Train using random frame selection per iteration"""
        for iteration in tqdm(range(self.num_iterations)):
            # Randomly select a frame
            frame_data = np.random.choice(self.frames_data)
            
            # Render
            render_output = render(
                frame_data['camera_info'],
                self.gaussian_model,
                self.pipeline_params,
                self.background_color
            )
            
            if render_output is None:
                continue
                
            rendered_image = render_output["render"]
            viewspace_points = render_output["viewspace_points"]
            visibility_filter = render_output["visibility_filter"]
            
            # Calculate loss
            target_image = frame_data['img'].permute(2, 0, 1)
            loss = torch.nn.functional.l1_loss(rendered_image, target_image)
            
            # Optimization step
            self.gaussian_model.optimizer.zero_grad()
            loss.backward()
            self.gaussian_model.optimizer.step()
            
            # Densification and pruning
            self.gaussian_model.add_densification_stats(viewspace_points, visibility_filter)
            self.gaussian_model.densify_and_prune(
                max_grad=0.01, min_opacity=0.005, extent=1.0, max_screen_size=10
            )
            
            # Save checkpoints
            if iteration % 200 == 0:
                self.save_checkpoint(iteration, rendered_image)
                
    def save_checkpoint(self, iteration, rendered_image):
        """Save model and rendered image"""
        # Save rendered image
        rendered_image_cpu = rendered_image.cpu().detach().numpy().transpose(1, 2, 0)
        rendered_image_cpu = (rendered_image_cpu * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(self.save_path, f"render_{iteration}.png"), rendered_image_cpu)
        
        # Save model
        self.gaussian_model.save_ply(os.path.join(self.save_path, f"model_{iteration}.ply"))

def main():
    config = Config()
    trainer = MultiFrameTrainer(config, 
                              num_frames=5, 
                              frame_skip=100, 
                              visualize_init=True)  # Enable visualization
    trainer.load_frames()
    trainer.initialize_gaussians()
    trainer.train()

if __name__ == "__main__":
    main()
