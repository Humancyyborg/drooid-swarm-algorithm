import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.octomap_creation import OctTree


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims = [10, 10, 10],
                 quad_size=0.046, size=0.0, resolution=0.05, inf_height=True):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.room_dims = room_dims
        self.octree = OctTree(obstacle_size=1.0, room_dims=room_dims, resolution=resolution, inf_height=inf_height)

    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.])):
        self.octree.reset()
        self.octree.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obstobs = []
        
        for quad in quads_pos:
            obstobs.append(self.octree.getSurround(quad))

        obs = np.concatenate((obs, obstobs), axis=1)

        return obs


    def step(self, obs=None, quads_pos=None):
        obstobs = []

        for quad in quads_pos:
            obstobs.append(self.octree.getSurround(quad))

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

    def closest_obstacle(self, pos):
        minimum = self.room_dims[0]+1
        closest = [0, 0, 0]
        for obstacle in self.octree.pos_arr:
            if np.linalg.norm(obstacle - pos) < minimum:
                minimum = np.linalg.norm(obstacle - pos)
                closest = obstacle
        return closest
