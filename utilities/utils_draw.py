"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import cv2
import torch
import random
import string
from utils_delaunay import draw_simplicies_on_image


# draw a list of points with different random colors on a input image 
def draw_points(img, pts, radius=5): 
    if img.ndim < 3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for pt in pts:
        color = tuple(np.random.randint(0,255,3).tolist())
        pt = tuple(map(int, pt))
        img = cv2.circle(img,tuple(pt),radius,color,-1)
    return img    


# draw corresponding points with the same random color on two separate images
def draw_points2(img1, img2, pts1, pts2, radius=5): 
    if img1.ndim < 3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if img2.ndim < 3:        
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))
        img1 = cv2.circle(img1,tuple(pt1),radius,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),radius,color,-1)
    return img1,img2    


# draw lines on a image; line_edges is assumed to be a list of 2D img points
def draw_lines(img, line_edges, pts=None, radius=5):
    pt = None 
    for i,l in enumerate(line_edges):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = l[0]
        x1,y1 = l[1]
        img = cv2.line(img, (int(x0),int(y0)), (int(x1),int(y1)), color,1)
        if pts is not None: 
            pt = tuple(map(int, pts[i])) 
            img = cv2.circle(img,pt,radius,color,-1)
    return img


# combine two images horizontally
def combine_images_horizontally(img1, img2): 
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)    
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)                     
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[:h2, w1:w1+w2,:3] = img2
    return img3 


# combine two images vertically
def combine_images_vertically(img1, img2): 
    if img1.ndim<=2:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)    
    if img2.ndim<=2:
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)                     
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img3 = np.zeros((h1+h2, max(w1, w2),3), np.uint8)
    img3[:h1, :w1,:3] = img1
    img3[h1:h1+h2,:w2,:3] = img2
    return img3 


# draw features matches (images are combined horizontally)
# input:
# - kps1 = [Nx2] array of keypoint coordinates 
# - kps2 = [Nx2] array of keypoint coordinates 
# - kps1_sizes = [Nx1] array of keypoint sizes 
# - kps2_sizes = [Nx1] array of keypoint sizes 
# output: drawn image 
def draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_horizontally(img1,img2)    
    h1,w1 = img1.shape[:2]    
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size        
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i] 
        size2 = kps2_sizes[i]    
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)    # optic flow style         
        cv2.line(img3, (a,b),(c+w1,d), color, 1)  # join corrisponding points 
        cv2.circle(img3,(a,b),2, color,-1)   
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=int(size1), thickness=1)  # draw keypoint size as a circle 
        cv2.circle(img3,(c+w1,d),2, color,-1) 
        cv2.circle(img3,(c+w1,d), color=(0, 255, 0), radius=int(size2), thickness=1)  # draw keypoint size as a circle  
    return img3    


# draw features matches (images are combined vertically)
# input:
# - kps1 = [Nx2] array of keypoint coordinates 
# - kps2 = [Nx2] array of keypoint coordinates 
# - kps1_sizes = [Nx1] array of keypoint sizes 
# - kps2_sizes = [Nx1] array of keypoint sizes 
# output: drawn image 
def draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None):
    img3 = combine_images_vertically(img1,img2) 
    h1,w1 = img1.shape[:2]           
    N = len(kps1)
    default_size = 2
    if kps1_sizes is None:
        kps1_sizes = np.ones(N,dtype=np.int32)*default_size
    if kps2_sizes is None:
        kps2_sizes = np.ones(N,dtype=np.int32)*default_size        
    for i,pts in enumerate(zip(kps1, kps2)):
        p1, p2 = np.rint(pts).astype(int)
        a,b = p1.ravel()
        c,d = p2.ravel()
        size1 = kps1_sizes[i] 
        size2 = kps2_sizes[i]    
        color = tuple(np.random.randint(0,255,3).tolist())
        #cv2.line(img3, (a,b),(c,d), color, 1)      # optic flow style   
        cv2.line(img3, (a,b),(c,d+h1), color, 1)   # join corrisponding points 
        cv2.circle(img3,(a,b),2, color,-1)   
        cv2.circle(img3,(a,b), color=(0, 255, 0), radius=int(size1), thickness=1)  # draw keypoint size as a circle 
        cv2.circle(img3,(c,d+h1),2, color,-1) 
        cv2.circle(img3,(c,d+h1), color=(0, 255, 0), radius=int(size2), thickness=1)  # draw keypoint size as a circle  
    return img3   


# draw features matches (images are combined horizontally)
# input:
# - kps1 = [Nx2] array of keypoint coordinates 
# - kps2 = [Nx2] array of keypoint coordinates 
# - kps1_sizes = [Nx1] array of keypoint sizes 
# - kps2_sizes = [Nx1] array of keypoint sizes 
# output: drawn image 
def draw_feature_matches(img1, img2, kps1, kps2, kps1_sizes=None, kps2_sizes=None, horizontal=True):
    if horizontal: 
        return draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes)    
    else:
        return draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes)


def draw_random_lines(img,N=200):
    lineType = 8
    (h, w) = img.shape[:2]    
    for i in range(N):
        pt1x, pt2x = np.random.randint( -0.5*w, w*1.5, 2)
        pt1y, pt2y = np.random.randint( -0.5*h, h*1.5, 2)
        color = tuple(np.random.randint(0,255,3).tolist())
        thickness = np.random.randint(1, 10)
        cv2.line(img, (pt1x,pt1y), (pt2x,pt2y), color, thickness, lineType)
        
        
def draw_random_rects(img,N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    for i in range(N):
        pt1x, pt2x = np.random.randint( 0, w, 2)
        pt1y, pt2y = np.random.randint( 0, h, 2)
        color = tuple(np.random.randint(0,255,3).tolist())        
        thickness = max(np.random.randint(-3, 10),-1)        
        cv2.rectangle(img, (pt1x,pt1y), (pt2x,pt2y), color, thickness, lineType)


def draw_torch_image(img):
    img = img.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    cv2.imshow("Image", img)
    cv2.waitKey(20)


def visualize_matches(img0, img1, kpts0, kpts1, matches, color=(0, 255, 0), thickness=2, radius=3,  add_text = False):
    """
    Visualizes keypoint matches between two images.

    Args:
        img0: The first image (numpy array, HxWxC or HxW).
        img1: The second image (numpy array, HxWxC or HxW).
        kpts0: Keypoints in the first image (torch.Tensor or numpy array, Nx2).
        kpts1: Keypoints in the second image (torch.Tensor or numpy array, Nx2).
        matches: A tensor/array of shape (M, 2) where each row contains indices (i, j) 
                 indicating that kpts0[i] matches kpts1[j].
        color:  Color of the lines and circles (B, G, R). Default: Green.
        thickness: Thickness of the connecting lines.
        radius: Radius of the keypoint circles.
        add_text : to add the text or not.

    Returns:
        output_img:  A combined image with matches visualized.
    """

    # Ensure images are numpy arrays
    if isinstance(img0, torch.Tensor):
        img0 = img0.permute(1, 2, 0).cpu().numpy()
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    
    # Convert grayscale to color if necessary
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    img0 = np.clip(img0 * 255, 0, 255).astype(np.uint8)
    img1 = np.clip(img1 * 255, 0, 255).astype(np.uint8)
    

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Stack images horizontally
    output_img = np.hstack([img0, img1])

    # Convert keypoints to numpy arrays and to integer pixel coordinates
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    if isinstance(matches, torch.Tensor):
        matches = matches.cpu().numpy()

    kpts0 = kpts0.astype(int)
    kpts1 = kpts1.astype(int)
    
    # Draw matches
    for i, j in matches:
        pt0 = tuple(kpts0[i])  # First image coordinates
        pt1 = tuple(kpts1[j] + np.array([w0, 0]))  # Second image, offset by width of first image

        cv2.circle(output_img, pt0, radius, color, thickness)
        cv2.circle(output_img, pt1, radius, color, thickness)
        cv2.line(output_img, pt0, pt1, color, thickness)

        if add_text:
            # Add text to the key points, more parameters can be added to make them look better.
            cv2.putText(output_img, str(i), pt0, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Add text to pt0
            cv2.putText(output_img, str(j), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # Add text to pt

    return output_img

def draw_random_ellipses(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]  
    axis_ext = w*0.1  
    for i in range(N):
        cx = np.random.randint( 0, w )
        cy = np.random.randint( 0, h )
        width, height = np.random.randint(0, axis_ext, 2)      
        angle = np.random.randint(0, 180)
        color = tuple(np.random.randint(0,255,3).tolist())        
        thickness = np.random.randint(-1, 9)   
        cv2.ellipse(img, (cx,cy), (width,height), angle, angle - 100, angle + 200, color, thickness, lineType)


def draw_random_polylines(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    num_pts = 3
    pts = np.zeros((num_pts,2),dtype=np.int32)
    for i in range(N):
        pts[:,0] = np.random.randint( 0, w, num_pts)
        pts[:,1] = np.random.randint( 0, h, num_pts)
        color = tuple(np.random.randint(0,255,3).tolist())        
        thickness = np.random.randint(1, 10)                   
        cv2.polylines(img, [pts], True, color, thickness, lineType)
        
        
def draw_random_polygons(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    num_pts = 3
    pts = np.zeros((num_pts,2),dtype=np.int32)
    for i in range(N):
        pts[:,0] = np.random.randint( 0, w, num_pts)
        pts[:,1] = np.random.randint( 0, h, num_pts)
        color = tuple(np.random.randint(0,255,3).tolist())          
        cv2.fillPoly(img, [pts], color, lineType)        


def draw_random_circles(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    radius_ext = w*0.1
    for i in range(N):
        cx = np.random.randint( 0, w )
        cy = np.random.randint( 0, h )
        color = tuple(np.random.randint(0,255,3).tolist())    
        radius = np.random.randint( 0, radius_ext)        
        thickness = np.random.randint(-1, 9)           
        cv2.circle(img, (cx,cy), radius, color, thickness, lineType )


def draw_random_text(img, N=100):
    lineType = 8
    (h, w) = img.shape[:2]    
    for i in range(N):
        cx = np.random.randint( 0, w )
        cy = np.random.randint( 0, h )
        random_char = random.choice(string.ascii_letters)
        font_face = np.random.randint( 0, 8 )
        scale = np.random.randint(0,5)+0.1
        color = tuple(np.random.randint(0,255,3).tolist())     
        thickness = np.random.randint(1, 10)                 
        cv2.putText(img, random_char, (cx,cy), font_face, scale, color, thickness, lineType);
    
    
def draw_random_img(shape): 
    #img_background = np.zeros(shape,dtype=np.uint8)           
    img_background = np.random.randint(255, size=shape,dtype=np.uint8)
    draw_random_rects(img_background)    
    draw_random_ellipses(img_background)                    
    draw_random_lines(img_background)           
    draw_random_polylines(img_background)    
    draw_random_polygons(img_background)   
    draw_random_circles(img_background)
    draw_random_text(img_background)
    img_background = cv2.GaussianBlur(img_background,ksize=(0,0),sigmaX=1)
    return img_background  


def visualize_matched_kps(prev_frame, cur_frame, idxs_ref, idxs_cur, fraction=1):
    # Visualize the common keypoints on current frame and previous frame - stack them side by side and draw lines connecting the keypoints
    print("Visualizing {} keypoints between frames.".format(len(idxs_ref)))
    stacked_image = np.hstack((prev_frame.img, cur_frame.img))
    num_keypoints = len(idxs_ref)
    selected_indices = random.sample(range(num_keypoints), int(fraction * num_keypoints))
    for idx in selected_indices:
        kp_ref = prev_frame.kpsu[idxs_ref[idx]]
        kp_cur = cur_frame.kpsu[idxs_cur[idx]]
        # Draw the stacked image
        cv2.circle(stacked_image, (int(kp_ref[0]), int(kp_ref[1])), 1, (0, 255, 0), -1)
        cv2.circle(stacked_image, (int(kp_cur[0] + prev_frame.img.shape[1]), int(kp_cur[1])), 1, (0, 255, 0), -1)
        cv2.line(stacked_image, (int(kp_ref[0]), int(kp_ref[1])), (int(kp_cur[0] + prev_frame.img.shape[1]), int(kp_cur[1])), np.random.randint(0, 255, 3).tolist(), 1)
    cv2.imshow("Stacked Image", stacked_image)
    cv2.waitKey(2)

def visualize_common_simplicies(curr_img, prev_img, curr_dict, prev_dict):
    curr_delaunay_img = draw_simplicies_on_image(curr_img, curr_dict)
    prev_delaunay_img = draw_simplicies_on_image(prev_img, prev_dict)
    stacked_image = np.hstack((prev_delaunay_img, curr_delaunay_img))
    cv2.imshow("Common Simplicies", stacked_image)
    cv2.waitKey(2)

    


    
# def visualize_matched_edges(prev_frame, cur_frame, idxs_ref, idxs_cur, common_edges_overall, fraction=0.05):
#     """
#     Visualizes edges (connections between keypoints) that are common between the current and previous frames.
    
#     Args:
#         prev_frame: Previous frame object containing keypoints and image.
#         cur_frame: Current frame object containing keypoints and image.
#         idxs_ref: Indices of keypoints in the previous frame that are matched.
#         idxs_cur: Indices of keypoints in the current frame that are matched.
#         common_edges_overall: List of edges (tuples) representing connections between common keypoints.
#         fraction: Fraction of edges to visualize (default: 0.05).
#     """
#     # Stack the previous and current images side by side
#     stacked_image = np.hstack((prev_frame.img, cur_frame.img))
    
#     if len(common_edges_overall) == 0:
#         print("No common edges to visualize.")
#         return
    
#     # Randomly select a subset of edges based on the specified fraction
#     num_edges = len(common_edges_overall)
#     selected_edges = random.sample(common_edges_overall, max(1, int(fraction * num_edges)))
    
#     prev_width = prev_frame.img.shape[1]  # Width of the previous image for offsetting current frame
    
#     for edge in selected_edges:
#         i, j = edge  # Indices in the common keypoints list
        
#         # Retrieve keypoints from previous frame using idxs_ref
#         prev_kp1 = tuple(map(int, prev_frame.kpsu[idxs_ref[i]]))
#         prev_kp2 = tuple(map(int, prev_frame.kpsu[idxs_ref[j]]))
        
#         # Retrieve keypoints from current frame using idxs_cur and adjust x-coordinate for stacking
#         cur_kp1 = tuple(map(int, cur_frame.kpsu[idxs_cur[i]]))
#         cur_kp2 = tuple(map(int, cur_frame.kpsu[idxs_cur[j]]))
#         cur_kp1_right = (cur_kp1[0] + prev_width, cur_kp1[1])
#         cur_kp2_right = (cur_kp2[0] + prev_width, cur_kp2[1])
        
#         # Generate a random color for consistent visualization across frames
#         color = tuple(np.random.randint(0, 255, 3).tolist())
        
#         # Draw edges in the previous frame (left side)
#         cv2.line(stacked_image, prev_kp1, prev_kp2, color, 1)
        
#         # Draw edges in the current frame (right side)
#         cv2.line(stacked_image, cur_kp1_right, cur_kp2_right, color, 1)
        
#         # Draw circles at keypoints and connecting lines across frames (optional)
#         cv2.circle(stacked_image, prev_kp1, 2, color, -1)
#         cv2.circle(stacked_image, prev_kp2, 2, color, -1)
#         cv2.circle(stacked_image, cur_kp1_right, 2, color, -1)
#         cv2.circle(stacked_image, cur_kp2_right, 2, color, -1)
#         cv2.line(stacked_image, prev_kp1, cur_kp1_right, color, 1)
#         cv2.line(stacked_image, prev_kp2, cur_kp2_right, color, 1)
    
#     cv2.imshow("Matched Edges Between Frames", stacked_image)
#     cv2.waitKey(2)


def visualize_matched_edges(prev_frame, cur_frame, idxs_ref, idxs_cur, common_edges_overall, fraction=1, scale_factor=2, line_thickness=3):
    """
    Visualizes edges (connections between keypoints) that are common between the current and previous frames,
    without drawing connecting lines between the two images. The image is resized for better visibility,
    and the edge thickness is increased.

    Args:
        prev_frame: Previous frame object containing keypoints and image.
        cur_frame: Current frame object containing keypoints and image.
        idxs_ref: Indices of keypoints in the previous frame that are matched.
        idxs_cur: Indices of keypoints in the current frame that are matched.
        common_edges_overall: List of edges (tuples) representing connections between common keypoints.
        fraction: Fraction of edges to visualize (default: 0.05).
        scale_factor: Factor to scale the image size (default: 2).
        line_thickness: Thickness of the lines used to draw edges (default: 2).
    """
    # Stack the previous and current images side by side
    stacked_image = np.hstack((prev_frame.img, cur_frame.img))
    
    if len(common_edges_overall) == 0:
        print("No common edges to visualize.")
        return
    
    # Randomly select a subset of edges based on the specified fraction
    num_edges = len(common_edges_overall)
    selected_edges = random.sample(common_edges_overall, max(1, int(fraction * num_edges)))
    
    prev_width = prev_frame.img.shape[1]  # Width of the previous image for offsetting current frame
    
    # Resize the stacked image to make it bigger
    stacked_image = cv2.resize(stacked_image, (stacked_image.shape[1] * scale_factor, stacked_image.shape[0] * scale_factor))
    
    # Scale factor for keypoints
    scale = (scale_factor, scale_factor)
    
    print("Visualizing {} edges between frames.".format(len(selected_edges)))
    
    for edge in selected_edges:
        # Generate a random color for consistent visualization across frames
        color = tuple(np.random.randint(0, 255, 3).tolist())
        i, j = edge  # Indices in the common keypoints list
        
        # Retrieve keypoints from previous frame using idxs_ref
        prev_kp1 = tuple(map(int, prev_frame.kpsu[idxs_ref[i]]))
        prev_kp2 = tuple(map(int, prev_frame.kpsu[idxs_ref[j]]))
        
        # Retrieve keypoints from current frame using idxs_cur and adjust x-coordinate for stacking
        cur_kp1 = tuple(map(int, cur_frame.kpsu[idxs_cur[i]]))
        cur_kp2 = tuple(map(int, cur_frame.kpsu[idxs_cur[j]]))
        
        # Apply scale to keypoints for resized image
        prev_kp1 = (int(prev_kp1[0] * scale[0]), int(prev_kp1[1] * scale[1]))
        prev_kp2 = (int(prev_kp2[0] * scale[0]), int(prev_kp2[1] * scale[1]))
        cur_kp1 = (int(cur_kp1[0] * scale[0]), int(cur_kp1[1] * scale[1]))
        cur_kp2 = (int(cur_kp2[0] * scale[0]), int(cur_kp2[1] * scale[1]))

        # Offset the current keypoints' x-coordinate based on the width of the previous image
        cur_kp1_right = (cur_kp1[0] + prev_width * scale_factor, cur_kp1[1])
        cur_kp2_right = (cur_kp2[0] + prev_width * scale_factor, cur_kp2[1])
        
        # Draw thicker edges in the previous frame (left side)
        cv2.line(stacked_image, prev_kp1, prev_kp2, color, line_thickness)
        
        # Draw thicker edges in the current frame (right side)
        cv2.line(stacked_image, cur_kp1_right, cur_kp2_right, color, line_thickness)
        
        # Draw circles at keypoints (optional)
        cv2.circle(stacked_image, prev_kp1, 2, color, -1)
        cv2.circle(stacked_image, prev_kp2, 2, color, -1)
        cv2.circle(stacked_image, cur_kp1_right, 2, color, -1)
        cv2.circle(stacked_image, cur_kp2_right, 2, color, -1)

    # Display the enlarged image
    cv2.imshow("Matched Edges Between Frames", stacked_image)
    cv2.waitKey(2)