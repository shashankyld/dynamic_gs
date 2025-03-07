import numpy as np
import open3d as o3d

class PC_Map():
    """ CLASS TO STORE POINT CLOUD MAP DATA """
    def __init__(self):
        self.map = {}

    def add(self, img_id, pointcloud):
        self.map[img_id] = pointcloud

    def get_pointcloud(self, img_id):
        return self.map[img_id]
    
    def get_full_pointcloud(self, fraction = 0.5, voxel_size=0.2):
        pcd = o3d.geometry.PointCloud()
        count = 0
        length = len(self.map)
        valid_counts = [i for i in range(length) if i % int(length * fraction) == 0]
        for key in valid_counts:
            count += 1
            # Take only a fraction of the point clouds with even distribution
            pcd += self.map[key]
        pcd.voxel_down_sample(voxel_size=voxel_size)
        return pcd
    
    

        

    
