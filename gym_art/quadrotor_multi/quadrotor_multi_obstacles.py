import numpy as np

from gym_art.quadrotor_multi.octomap_creation import OctTree
from gym_art.quadrotor_multi.quad_utils import EPS


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), resolution=0.05, obstacle_size=1.0):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.room_dims = room_dims
        self.obstacle_size = obstacle_size
        self.resolution = resolution
        self.octree = OctTree(obstacle_size=self.obstacle_size, room_dims=room_dims,
                              resolution=resolution)
        self.closest_obst_dist = []

    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.])):
        self.octree.reset()
        self.octree.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obs = self.concate_obst_obs(quads_pos=quads_pos, obs=obs)
        return obs

    def step(self, obs=None, quads_pos=None):
        obs = self.concate_obst_obs(quads_pos=quads_pos, obs=obs)
        return obs

    def concate_obst_obs(self, quads_pos, obs):
        obst_obs = []

        for quad in quads_pos:
            surround_obs = self.octree.get_surround(quad)
            approx_part = np.random.uniform(low=-1.0 * self.resolution, high=0.0, size=surround_obs.shape)

            surround_obs += approx_part
            surround_obs = np.maximum(surround_obs, 0.0)
            obst_obs.append(surround_obs)

        obst_obs = np.array(obst_obs)

        # Extract closest obst
        self.extract_closest_obst_dist(obst_obs=obst_obs)

        # Add noise to obst_obs
        noise_part = np.random.normal(loc=0, scale=0.01, size=obst_obs.shape)
        obst_obs += noise_part

        obst_obs = np.maximum(obst_obs, 0.0)
        obs = np.concatenate((obs, obst_obs), axis=1)

        return obs

    def extract_closest_obst_dist(self, obst_obs):
        self.closest_obst_dist = []
        for item in obst_obs:
            tmp_item = item.flatten()
            center_idx = int(len(tmp_item) - 1 / 2)
            center_dist = tmp_item[center_idx]
            self.closest_obst_dist.append(center_dist)

        self.closest_obst_dist = np.array(self.closest_obst_dist)

    def collision_detection(self):
        drone_collision = np.where(self.closest_obst_dist < 0.06 + EPS)[0]
        return drone_collision

    def closest_obstacle(self, pos):
        rel_dist = np.linalg.norm(self.octree.pos_arr[:, :2] - pos[:2], axis=1)
        closest_index = np.argmin(rel_dist)
        closest = self.octree.pos_arr[closest_index]
        return closest
