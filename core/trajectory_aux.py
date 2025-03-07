import numpy as np

class Trajectory():
    """ CLASS TO STORE TRAJECTORY DATA """
    def __init__(self):
        self.trajectory = {}

    def add(self, img_id, pose):
        self.trajectory[img_id] = pose

    def get_pose(self, img_id):
        return self.trajectory[img_id]
    
    def get_inv_pose(self, img_id):
        return np.linalg.inv(self.trajectory[img_id])

    def get_relative_pose(self, img_id1, img_id2):
        return np.linalg.inv(self.trajectory[img_id1]) @ self.trajectory[img_id2]
    
    def get_relative_inv_pose(self, img_id1, img_id2):
        return np.linalg.inv(self.get_relative_pose(img_id1, img_id2))
