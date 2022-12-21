import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.octomap_creation import OctTree


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims = [10, 10, 10],
                 quad_size=0.046, size=0.0, resolution=0.05, inf_height=True):
        #TODO Fix this
        self.num_obstacles = 4
        self.obstacles = []
        
        self.octree = OctTree(obstacle_size=1.0, room_dims=room_dims, resolution=resolution, inf_height=inf_height)
        '''self.octree.generate_obstacles(num_obstacles=num_obstacles)
        self.octree.pos_arr
        self.octree.mark_octree()
        self.octree.generateSDF()'''

    #Test shapes of obs, test collisions
    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.])):
        self.octree.reset()
        self.octree.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obstobs = []
        
        for quad in quads_pos:
            obstobs.append(self.octree.getSurround(quad))

        #obstobs = [np.ones(27)] * 8
        obs = np.concatenate((obs, obstobs), axis=1)

        return obs


    #Returns observations
    def step(self, obs=None, quads_pos=None):
        obstobs = []

        for quad in quads_pos:
            obstobs.append(self.octree.getSurround(quad))

        #obstobs = [np.ones(27)] * 8

        obs = np.concatenate((obs, obstobs), axis=1)

        return obs

    def collision_detection(self, pos_quads=None):

        collision_matrix = [0] * len(pos_quads)
        
        for i, quad in enumerate(pos_quads):
            curr = self.octree.SDFDist(quad)
            # TODO
            if curr < 0.05:
                collision_matrix[i] = 1
        
        return collision_matrix
