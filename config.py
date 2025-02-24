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

import sys
if sys.version_info[0] != 3:
    print("This script requires Python 3")
    exit()
    
import os
import yaml
import numpy as np
from utilities.utils_sys import Printer, locally_configure_qt_environment
import math
import slam_parameters 


# N.B.: this file must stay in the root folder of the repository 


# get the folder location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Class for getting libs settings (from config.yaml) and camera settings from a yaml file 
class Config(object):
    '''
    Config is used for getting libs settings (from config.yaml) and camera settings from a yaml file 
    '''
    def __init__(self):
        self.root_folder = __location__
        self.config_file = 'config.yaml'
        self.config_file_path = __location__ + '/' + self.config_file
        #print(f'root folder: {self.root_folder}, config file path: {self.config_file_path}')
        self.config = yaml.load(open(self.config_file_path, 'r'), Loader=yaml.FullLoader)
        self.cam_settings = None
        self.cam_stereo_settings = None
        self.feature_manager_settings = None
        self.dataset_settings = None
        self.dataset_type = None
        self.sensor_type = None
        self.start_frame_id = slam_parameters.SlamParameters.kStartingFrameIdx
        self.end_frame_id = slam_parameters.SlamParameters.kEndingFrameIdx
        self.NumFramesAway = slam_parameters.SlamParameters.kNumFramesAway
        self.num_features_to_extract = slam_parameters.SlamParameters.kNumFeatures
        self.ShowDebugImages = slam_parameters.SlamParameters.kShowDebugImages
        self.NumLocalKFs = slam_parameters.SlamParameters.kNumLocalKFs
        
        # New SLAM Parameters
        self.min_distance_between_kfs = slam_parameters.SlamParameters.MIN_DISTANCE_BETWEEN_KEYFRAMES
        self.min_rotation_between_kfs = slam_parameters.SlamParameters.MIN_ROTATION_BETWEEN_KEYFRAMES
        self.max_frames_between_kfs = slam_parameters.SlamParameters.MAX_FRAMES_BETWEEN_KEYFRAMES
        self.min_inlier_threshold = slam_parameters.SlamParameters.MIN_INLIER_THRESHOLD
        self.min_keyframe_matches = slam_parameters.SlamParameters.MIN_KEYFRAME_MATCHES

        self.get_dataset_settings()
        self.get_cam_settings()
        self.get_feature_manager_settings()



    # get dataset settings
    def get_dataset_settings(self):
        self.dataset_type = self.config['DATASET']['type']
        self.dataset_settings = self.config[self.dataset_type]
        self.sensor_type = self.dataset_settings['sensor_type'].lower()
        self.dataset_path = self.dataset_settings['base_path']
        self.dataset_settings['base_path'] = os.path.join( __location__, self.dataset_path)
        self.gs_opt_params = self.dataset_settings['gs_opt_params']
        self.pipeline_params = self.dataset_settings['pipeline_params']

    # get camera settings
    def get_cam_settings(self):
        self.cam_settings = None


        self.cam_settings_filepath = __location__ + '/' + self.config[self.dataset_type]['settings']
        if self.sensor_type == 'stereo':
            if 'settings_stereo' in self.config[self.dataset_type]:
                self.cam_settings_filepath = __location__ + '/' + self.config[self.dataset_type]['settings_stereo']
                Printer.orange('Using stereo settings file: ' + self.cam_settings_filepath)
                print('------------------------------------')            
        if(self.cam_settings_filepath is not None):
            with open(self.cam_settings_filepath, 'r') as stream:
                try:
                    self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)
                    
    # get feature manager settings
    def get_feature_manager_settings(self):
        self.feature_manager_settings = None

        self.feature_manager_settings_filepath = __location__ + '/' + self.config[self.dataset_type]['settings']
        if(self.feature_manager_settings_filepath is not None):
            with open(self.feature_manager_settings_filepath, 'r') as stream:
                try:
                    self.feature_manager_settings = yaml.load(stream, Loader=yaml.FullLoader)
                except yaml.YAMLError as exc:
                    print(exc)                    


    # calibration matrix
    @property
    def K(self):
        if not hasattr(self, '_K'):
            fx = self.cam_settings['Camera.fx']
            cx = self.cam_settings['Camera.cx']
            fy = self.cam_settings['Camera.fy']
            cy = self.cam_settings['Camera.cy']
            self._K = np.array([[fx,  0, cx],
                                [ 0, fy, cy],
                                [ 0,  0,  1]])
        return self._K

    # inverse of calibration matrix
    @property
    def Kinv(self):
        if not hasattr(self, '_Kinv'):
            fx = self.cam_settings['Camera.fx']
            cx = self.cam_settings['Camera.cx']
            fy = self.cam_settings['Camera.fy']
            cy = self.cam_settings['Camera.cy']
            self._Kinv = np.array([[1/fx,    0, -cx/fx],
                                   [   0, 1/fy, -cy/fy],
                                   [   0,    0,      1]])
        return self._Kinv

    # distortion coefficients
    @property
    def DistCoef(self):
        if not hasattr(self, '_DistCoef'):
            k1 = self.cam_settings['Camera.k1']
            k2 = self.cam_settings['Camera.k2']
            p1 = self.cam_settings['Camera.p1']
            p2 = self.cam_settings['Camera.p2']
            k3 = 0
            if 'Camera.k3' in self.cam_settings:
                k3 = self.cam_settings['Camera.k3']
            self._DistCoef = np.array([k1, k2, p1, p2, k3])
            if self.sensor_type == 'stereo':
                self._DistCoef = np.array([0, 0, 0, 0, 0])
                Printer.orange('WARNING: Using stereo camera, images are automatically rectified, and DistCoef is set to [0,0,0,0,0]')
        return self._DistCoef
    
    # baseline times fx
    @property
    def bf(self):
        if not hasattr(self, '_bf'):
            self._bf = self.cam_settings['Camera.bf']
        return self._bf

    # camera width
    @property
    def width(self):
        if not hasattr(self, '_width'):
            self._width = self.cam_settings['Camera.width']
        return self._width

    # camera height
    @property
    def height(self):
        if not hasattr(self, '_height'):
            self._height = self.cam_settings['Camera.height']
        return self._height
    
    # camera fps
    @property
    def fps(self):
        if not hasattr(self, '_fps'):
            self._fps= self.cam_settings['Camera.fps']
        return self._fps    

    # depth factor
    @property
    def depth_factor(self):
        if not hasattr(self, '_depth_factor'):
            if 'Camera.DepthMapFactor' in self.cam_settings:
                self._depth_factor = self.cam_settings['Camera.DepthMapFactor']
            else:
                self._depth_factor = 1.0
        return self._depth_factor
    
    # depth threshold 
    @property
    def depth_threshold(self):
        if not hasattr(self, '_depth_threshold'):
            if 'Camera.ThDepth' in self.cam_settings:
                self._depth_threshold = self.cam_settings['Camera.ThDepth']
            else:
                self._depth_threshold = float('inf')
        return self._depth_threshold



    # stereo settings 
    @property
    def stereo_settings(self):
        if not hasattr(self, '_stereo_settings'):
            self._stereo_settings = None
            left, right = {}, {}
            if 'LEFT.D' in self.cam_settings:
                left_D = self.cam_settings['LEFT.D']
                left_D = np.array(left_D['data'],dtype=float).reshape(left_D['rows'], left_D['cols'])
                left['D'] = left_D
            if 'LEFT.K' in self.cam_settings:
                left_K = self.cam_settings['LEFT.K']
                left_K = np.array(left_K['data'],dtype=float).reshape(left_K['rows'], left_K['cols'])
                left['K'] = left_K
            if 'LEFT.R' in self.cam_settings:
                left_R = self.cam_settings['LEFT.R']
                left_R = np.array(left_R['data'],dtype=float).reshape(left_R['rows'], left_R['cols'])
                left['R'] = left_R
            if 'LEFT.P' in self.cam_settings:
                left_P = self.cam_settings['LEFT.P']
                left_P = np.array(left_P['data'],dtype=float).reshape(left_P['rows'], left_P['cols'])
                left['P'] = left_P
                
            if 'RIGHT.D' in self.cam_settings:
                right_D = self.cam_settings['RIGHT.D']
                right_D = np.array(right_D['data'],dtype=float).reshape(right_D['rows'], right_D['cols'])
                right['D'] = right_D
            if 'RIGHT.K' in self.cam_settings:
                right_K = self.cam_settings['RIGHT.K']
                right_K = np.array(right_K['data'],dtype=float).reshape(right_K['rows'], right_K['cols'])
                right['K'] = right_K
            if 'RIGHT.R' in self.cam_settings:
                right_R = self.cam_settings['RIGHT.R']
                right_R = np.array(right_R['data'],dtype=float).reshape(right_R['rows'], right_R['cols'])
                right['R'] = right_R 
            if 'RIGHT.P' in self.cam_settings:
                right_P = self.cam_settings['RIGHT.P']
                right_P = np.array(right_P['data'],dtype=float).reshape(right_P['rows'], right_P['cols'])
                right['P'] = right_P         
                   
            if len(left) > 0 and len(right) > 0:
                self._stereo_settings = {'left':left, 'right':right}
        #print(f'[config] stereo settings: {self._stereo_settings}')
        return self._stereo_settings

    @property
    def slam_params(self):
        """Get all SLAM parameters as a dictionary."""
        if not hasattr(self, '_slam_params'):
            self._slam_params = {
                'start_frame': self.start_frame_id,
                'end_frame': self.end_frame_id,
                'num_frames_away': self.NumFramesAway,
                'num_features': self.num_features_to_extract,
                'show_debug': self.ShowDebugImages,
                'num_local_kfs': self.NumLocalKFs,
                'min_distance_kf': self.min_distance_between_kfs,
                'min_rotation_kf': self.min_rotation_between_kfs,
                'max_frames_kf': self.max_frames_between_kfs,
                'min_inliers': self.min_inlier_threshold,
                'min_matches': self.min_keyframe_matches
            }
        return self._slam_params

   
if __name__ != "__main__":
    # we automatically read lib path when this file is called via 'import'
    cfg = Config()
