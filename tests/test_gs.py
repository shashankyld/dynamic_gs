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
from munch import Munch

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
gaussian_model = GaussianModel(sh_degree=sh_degree, config = config)
gaussian_model.init_lr(6.0)

# Setting up training parameters 
training_params = config.gs_opt_params
pipeline_params = config.pipeline_params
training_args = Munch(training_params)  # Convert dict to Munch
pipeline_args = Munch(pipeline_params)  # Convert dict to Munch 
# print("training_params", training_args)
gaussian_model.training_setup(training_args)

## TODO TODO TODO
# 2. Camera Info (replace with your actual camera parameters)
camera_info = get_camera_info_from_pyslam_dataloader_insteadofgs(dataset, ground_truth, img_id=0, config=config)


# Initialize the Gaussian model with the first frame (as you did before)
fused_point_cloud, features, scales, rots, opacities = gaussian_model.create_pcd_from_image(camera_info, depthmap=depth)
print("fused_point_cloud", fused_point_cloud)
print("features", features)
print("scales", len(scales))
print("rots", rots)
print("opacities", opacities)

gaussian_model.extend_from_pcd(fused_point_cloud, features, scales, rots, opacities, kf_id=0) # kf_id is just an example

print("Optimizer: ", gaussian_model.optimizer)

# 4. Training Loop
for iteration in tqdm(range(num_iterations)):
    # a. Render
    render_output = render(camera_info, gaussian_model, pipeline_args, background_color) # None is the pipe object.
    
    if render_output is None:
        continue # if there are no gaussians to render

    rendered_image = render_output["render"]
    viewspace_points = render_output["viewspace_points"]
    visibility_filter = render_output["visibility_filter"]

    # Visualize the rendered image
    print("Shape of render_output", rendered_image.shape)
    print("Type of render_output", type(rendered_image))
    print("Type of target image", type(img))
    print("Shape of the target image", img.shape)
    rendered_img_view = rendered_image.cpu().detach().numpy().transpose(1, 2, 0)
    
    # cv2.imshow("Rendered Image", rendered_img_view)
    # cv2.imshow("Target Image", img)
    # cv2.waitKey(1000)


    # b. Calculate Loss (example: L1 loss)
    target_image = torch.from_numpy(img).to(device).float() / 255.0  # Assuming img is a NumPy array
    target_image = target_image.permute(2, 0, 1)

    loss = torch.nn.functional.l1_loss(rendered_image, target_image)

    # c. Backpropagation
    gaussian_model.optimizer.zero_grad()
    loss.backward()

    # d. Optimization Step
    gaussian_model.optimizer.step()

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
        gaussian_model.save_ply(os.path.join(os.getcwd(), f"model_{iteration}.ply"))

# 5. (After training) Render final result or save model
final_render = render(camera_info, gaussian_model, None, background_color)["render"]
#... save final_render...
gaussian_model.save_ply("final_model.ply")

