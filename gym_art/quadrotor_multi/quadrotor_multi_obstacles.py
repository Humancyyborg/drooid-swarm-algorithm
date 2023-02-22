import numpy as np

from gym_art.quadrotor_multi.octomap_creation import OctTree
from gym_art.quadrotor_multi.utils.quad_obst_utils import calculate_obst_drone_proximity_penalties, \
    get_vel_omega_change_obst_collisions
from gym_art.quadrotor_multi.utils.quad_utils import EPS


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims=np.array([10, 10, 10]), resolution=0.05, obstacle_size=1.0,
                 collision_obst_falloff_radius=3.0, num_agents=8, rew_coeff=None, control_dt=0.01):
        # Pre-set
        self.num_obstacles = num_obstacles
        self.obstacles = []
        self.room_dims = room_dims
        self.obstacle_size = obstacle_size
        self.resolution = resolution
        self.closest_obst_dist = []
        self.num_agents = num_agents
        self.rew_coeff = rew_coeff
        self.control_dt = control_dt

        self.octree = OctTree(obstacle_size=self.obstacle_size, room_dims=room_dims,
                              resolution=resolution)

        # Collisions
        self.obst_quad_col_matrix = np.array([])
        self.prev_obst_quad_collisions = np.array([])
        self.obst_quad_collisions_per_episode = 0
        self.collision_obst_falloff_radius = collision_obst_falloff_radius

        # Reward
        self.curr_quad_col = np.array([])
        self.rew_obst_quad_collisions_raw = np.zeros(num_agents)

    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.]),
              rew_coeff=None):
        # Update
        self.rew_coeff = rew_coeff

        # Reset collisions
        self.obst_quad_col_matrix = np.array([])
        self.prev_obst_quad_collisions = np.array([])
        self.obst_quad_collisions_per_episode = 0

        self.octree.reset()
        self.octree.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obs = self.concate_obst_obs(quads_pos=quads_pos, obs=obs)
        return obs

    def step(self, obs, sense_positions):
        obs = self.concate_obst_obs(quads_pos=sense_positions, obs=obs)
        return obs

    def calculate_collision_info(self, rew_coeff):
        # Update
        self.rew_coeff = rew_coeff
        self.obst_quad_col_matrix = self.collision_detection()
        # We assume drone can only collide with one obstacle at the same time.
        # Given this setting, in theory, the gap between obstacles should >= 0.1 (drone diameter: 0.46*2 = 0.92)
        self.curr_quad_col = np.setdiff1d(self.obst_quad_col_matrix, self.prev_obst_quad_collisions)
        self.obst_quad_collisions_per_episode += len(self.curr_quad_col)

        self.prev_obst_quad_collisions = self.obst_quad_col_matrix

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
        drone_collision = np.where(np.logical_and(self.closest_obst_dist < 0.06 + EPS,
                                                  self.closest_obst_dist > -1.0 * EPS))[0]
        return drone_collision

    def closest_obstacle(self, pos):
        rel_dist = np.linalg.norm(self.octree.pos_arr[:, :2] - pos[:2], axis=1)
        closest_index = np.argmin(rel_dist)
        closest = self.octree.pos_arr[closest_index]
        return closest

    def perform_physical_interaction(self, real_positions, real_velocities):
        obstacle_poses = []
        for val in self.obst_quad_col_matrix:
            obstacle_poses.append(self.closest_obstacle(real_positions[val]))

        obstacle_poses = np.array(obstacle_poses)

        obst_velocities_change, obst_omegas_change = get_vel_omega_change_obst_collisions(
            num_agents=self.num_agents, obst_quad_col_matrix=self.obst_quad_col_matrix,
            real_positions=real_positions, real_velocities=real_velocities, obstacle_size=self.obstacle_size,
            obstacle_poses=obstacle_poses, col_coeff=self.rew_coeff["quadcol_obst_coeff"])

        return obst_velocities_change, obst_omegas_change

    def calculate_reward(self, real_positions):
        self.rew_obst_quad_collisions_raw = np.zeros(self.num_agents)

        # We assign penalties to the drones which collide with the obstacles
        # And obst_quad_last_step_unique_collisions only include drones' id
        if len(self.obst_quad_col_matrix) > 0:
            for i in self.curr_quad_col:
                self.rew_obst_quad_collisions_raw[i] = -1.0

        rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * self.rew_obst_quad_collisions_raw

        # Penalties for smallest distance between obstacles and drones
        # Only penalize the smallest instead of checking all nearby obstacles makes sense.
        # Since we don't want drones afraid of flying into obstacle dense zone.
        drone_obst_dists = np.array([self.octree.sdf_dist(real_positions[i]) for i in range(self.num_agents)])

        rew_obst_quad_proximity = -1.0 * calculate_obst_drone_proximity_penalties(
            distances=drone_obst_dists, dt=self.control_dt,
            penalty_fall_off=self.collision_obst_falloff_radius,
            max_penalty=self.rew_coeff["quadcol_bin_obst_smooth_max"],
            num_agents=self.num_agents,
        )

        return rew_collisions_obst_quad, rew_obst_quad_proximity
