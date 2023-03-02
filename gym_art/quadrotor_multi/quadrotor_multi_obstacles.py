import numpy as np

from gym_art.quadrotor_multi.octomap_creation import OctTree


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), resolution=0.05, obstacle_size=1.0, obst_shape="cube"):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.room_dims = room_dims
        self.obst_shape = obst_shape
        self.obstacle_size = obstacle_size
        self.octree = OctTree(obstacle_size=self.obstacle_size, room_dims=room_dims,
                              resolution=resolution, obst_shape=self.obst_shape)

    def reset(self, obs=None, quads_pos=None, pos_arr=None):
        self.octree.reset()
        self.octree.set_obst(pos_arr)

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
            if curr < 0.1 + 1e-5:
                drone_collision.append(i)

        return drone_collision

    def closest_obstacle(self, pos):
        rel_dist = np.linalg.norm(self.octree.pos_arr[:, :2] - pos[:2], axis=1)
        closest_index = np.argmin(rel_dist)
        closest = self.octree.pos_arr[closest_index]
        return closest
