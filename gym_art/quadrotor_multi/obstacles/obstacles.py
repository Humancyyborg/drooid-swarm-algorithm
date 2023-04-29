import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_surround_sdfs_radar_2d


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, obst_obs_type='octomap', obst_scan_range=np.pi,
                 obst_ray_num=4):
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = 0.1
        self.obst_obs_type = obst_obs_type
        self.obst_scan_range = obst_scan_range
        self.obst_ray_num = obst_ray_num

    def reset(self, obs, quads_pos, pos_arr, quads_vel):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))

        if self.obst_obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        elif self.obst_obs_type == 'radar':
            quads_sdf_obs = get_surround_sdfs_radar_2d(
                quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2], quad_vels=quads_vel[:, :2],
                obst_radius=self.obstacle_radius, scan_range=self.obst_scan_range, ray_num=self.obst_ray_num)
        else:
            raise NotImplementedError(f'self.obst_obs_type: {self.obst_obs_type} is not supported!')

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos, quads_vel):
        if self.obst_obs_type == 'octomap':
            quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        elif self.obst_obs_type == 'radar':
            quads_sdf_obs = get_surround_sdfs_radar_2d(
                quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2], quad_vels=quads_vel[:, :2],
                obst_radius=self.obstacle_radius, scan_range=self.obst_scan_range, ray_num=self.obst_ray_num)
        else:
            raise NotImplementedError(f'self.obst_obs_type: {self.obst_obs_type} is not supported!')

        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
