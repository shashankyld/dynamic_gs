import torch
import numpy as np
import open3d as o3d
import sys
sys.path.append("/home/shashank/Documents/UniBonn/thesis/GS/dynamic_gs/thirdparty/")
from tqdm import tqdm

from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.general_utils import build_rotation
from thirdparty.gaussian_splatting.utils.dataset_bridge import get_camera_info_from_pyslam_dataloader_insteadofgs


import os
import torch
import numpy as np
import cv2  # For image loading and manipulation
from thirdparty.gaussian_splatting.utils.camera_utils import Camera # If you have a CameraInfo class
from thirdparty.gaussian_splatting.utils.graphics_utils import getWorld2View2, BasicPointCloud
from thirdparty.gaussian_splatting.utils.sh_utils import RGB2SH
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from simple_knn._C import distCUDA2 # if you are using simple_knn

# FOR DATASET LOADER
from config import Config
from io_utils.dataset import dataset_factory, SensorType
from io_utils.ground_truth import groundtruth_factory
from config_parameters import Parameters  


config = Config()
dataset = dataset_factory(config)
ground_truth = groundtruth_factory(config.dataset_settings)
depth_scale = 1/5000 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_id = 0
img = dataset.getImage(img_id)
depth = dataset.getDepth(img_id) * depth_scale

sh_degree = 0 
# Width and height of the rendered image same as the input image
render_width = dataset.width
render_height = dataset.height
learning_rate = 0.001
num_iterations = 1000
background_color = torch.zeros(3).to(device)
fx = config.cam_settings['Camera.fx']
fy = config.cam_settings['Camera.fy']
cx = config.cam_settings['Camera.cx']
cy = config.cam_settings['Camera.cy']
R = np.eye(3)
T = np.zeros(3)

# 1. Create Gaussian Model
gaussian_model = GaussianModel(sh_degree=sh_degree)

## TODO TODO TODO
# 2. Camera Info (replace with your actual camera parameters)
camera_info = get_camera_info_from_pyslam_dataloader_insteadofgs(dataset, ground_truth, img_id=0, config=config)


# Initialize the Gaussian model with the first frame (as you did before)
fused_point_cloud, features, scales, rots, opacities = gaussian_model.create_pcd_from_image(camera_info)
gaussian_model.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id=0) # kf_id is just an example

# 3. Optimizer
optimizer = torch.optim.Adam(gaussian_model.parameters(), lr=learning_rate)

# 4. Training Loop
for iteration in tqdm(range(num_iterations)):
    # a. Render
    render_output = render(camera_info, gaussian_model, None, background_color) # None is the pipe object.

    if render_output is None:
        continue # if there are no gaussians to render

    rendered_image = render_output["render"]
    viewspace_points = render_output["viewspace_points"]
    visibility_filter = render_output["visibility_filter"]
    # b. Calculate Loss (example: L1 loss)
    target_image = torch.from_numpy(img).to(device).float() / 255.0  # Assuming img is a NumPy array
    loss = torch.nn.functional.l1_loss(rendered_image, target_image)

    # c. Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # d. Optimization Step
    optimizer.step()

    # e. Densification and Pruning (example - adapt as needed)
    gaussian_model.add_densification_stats(viewspace_points, visibility_filter)
    gaussian_model.densify_and_prune(max_grad=0.01, min_opacity=0.005, extent=1.0, max_screen_size=10)

    # f. Print or log loss
    print(f"Iteration: {iteration}, Loss: {loss.item()}")

    # g. (Optional) Save rendered images or model checkpoints
    if iteration % 10 == 0:  # Save every 10 iterations
        rendered_image_cpu = rendered_image.cpu().detach().numpy().transpose(1, 2, 0)
        rendered_image_cpu = (rendered_image_cpu * 255).astype(np.uint8)
        cv2.imwrite(f"rendered_{iteration}.png", rendered_image_cpu)
        gaussian_model.save_ply(f"model_{iteration}.ply")


# 5. (After training) Render final result or save model
final_render = render(camera_info, gaussian_model, None, background_color)["render"]
#... save final_render...
gaussian_model.save_ply("final_model.ply")

