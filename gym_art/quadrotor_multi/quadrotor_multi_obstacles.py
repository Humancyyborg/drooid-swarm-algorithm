import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.octomap_creation import OctTree


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), size=0.0, resolution=0.05):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.room_dims = room_dims
        self.octree = OctTree(obstacle_size=1.0, room_dims=room_dims, resolution=resolution)

    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.])):
        self.octree.reset()
        self.octree.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obst_obs = []

        for quad in quads_pos:
            obst_obs.append(self.octree.get_surround(quad))

        obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def step(self, obs=None, quads_pos=None):
        obst_obs = []

        for quad in quads_pos:
            obst_obs.append(self.octree.get_surround(quad))

        obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads=None):
        drone_collision = []

        for i, quad in enumerate(pos_quads):
            curr = self.octree.sdf_dist(quad)
            if curr < 0.05:
                drone_collision.append(i)

        return drone_collision

    def closest_obstacle(self, pos):
        rel_dist = np.linalg.norm(self.octree.pos_arr[:, :2] - pos[:2], axis=1)
        closest_index = np.argmin(rel_dist)
        closest = self.octree.pos_arr[closest_index]
        return closest
