import numpy as np

from gym_art.quadrotor_multi.utils.quad_neighbor_utils import add_neighborhood_obs, \
    calculate_collision_matrix, COLLISIONS_GRACE_PERIOD, calculate_drone_proximity_penalties, perform_downwash, \
    get_vel_omega_change_neighbor_collisions
from gym_art.quadrotor_multi.utils.quad_utils import SELF_OBS_REPR, NEIGHBOR_OBS


class MultiNeighbors:
    def __init__(self, obs_repr='xyz_vxyz_R_omega', obs_type='pos_vel', visible_neighbor_num=0, use_downwash=False,
                 collision_hitbox_radius=2., collision_falloff_radius=4., collision_smooth_max_penalty=10.,
                 num_agents=8, control_freq=100, rew_coeff=None, observation_space=None):
        # Pre-set
        self.num_agents = num_agents
        self.control_freq = control_freq
        self.control_dt = 1.0 / self.control_freq
        self.rew_coeff = rew_coeff

        # Obs
        obs_self_size = SELF_OBS_REPR[obs_repr]
        obs_size = NEIGHBOR_OBS[obs_type]
        self.obs_type = obs_type
        self.visible_neighbor_num = visible_neighbor_num

        self.clip_obs_length = visible_neighbor_num * obs_size
        self.clip_obs_low_box = observation_space.low[obs_self_size:obs_self_size + self.clip_obs_length]
        self.clip_obs_high_box = observation_space.high[obs_self_size:obs_self_size + self.clip_obs_length]

        # Collision
        # # Use to calculate unique collisions
        self.prev_drone_collisions = []
        self.curr_drone_collisions = []

        # # Total number of pairwise collisions / episode
        self.collisions_per_episode = 0
        # # Some collisions may happen because the quad-rotors get initialized on the collision course
        # # if we wait a couple of seconds, then we can eliminate all the collisions that happen due to initialization
        # # this is the actual metric that we want to minimize
        self.collisions_after_settle = 0

        # Reward
        self.collision_hitbox_radius = collision_hitbox_radius
        self.collision_falloff_radius = collision_falloff_radius
        self.collision_smooth_max_penalty = collision_smooth_max_penalty
        self.distance_matrix = np.array([])

        # Replay buffer
        self.last_step_unique_collisions = np.array([])

        # Down wash
        self.use_downwash = use_downwash

        # Aux for render
        self.drone_col_matrix = np.array([])

    def reset(self, obs, sense_positions, sense_velocities, goals, rew_coeff):
        # Update
        self.rew_coeff = rew_coeff

        # Concatenate observations of neighbor drones
        obs = add_neighborhood_obs(
            obs=obs, swarm_obs=self.obs_type, num_agents=self.num_agents, positions=sense_positions,
            velocities=sense_velocities, num_use_neighbor_obs=self.visible_neighbor_num, goals=goals,
            clip_neighbor_space_min_box=self.clip_obs_low_box,
            clip_neighbor_space_max_box=self.clip_obs_high_box)

        # Reset collision variables
        self.prev_drone_collisions = []
        self.collisions_per_episode = 0
        self.collisions_after_settle = 0

        return obs

    def step(self, obs, sense_positions, sense_velocities, goals):
        # Concatenate observations of neighbor drones
        obs = add_neighborhood_obs(
            obs=obs, swarm_obs=self.obs_type, num_agents=self.num_agents, positions=sense_positions,
            velocities=sense_velocities, num_use_neighbor_obs=self.visible_neighbor_num, goals=goals,
            clip_neighbor_space_min_box=self.clip_obs_low_box,
            clip_neighbor_space_max_box=self.clip_obs_high_box)

        return obs

    def calculate_collision_info(self, rew_coeff, real_positions, cur_tick):
        # Update
        self.rew_coeff = rew_coeff

        # Calculate info for collision and reward
        self.drone_col_matrix, self.curr_drone_collisions, self.distance_matrix = \
            calculate_collision_matrix(positions=real_positions, hitbox_radius=self.collision_hitbox_radius)

        self.last_step_unique_collisions = np.setdiff1d(self.curr_drone_collisions, self.prev_drone_collisions)

        # collision between 2 drones counts as a single collision
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0 and cur_tick >= COLLISIONS_GRACE_PERIOD * self.control_freq:
            self.collisions_after_settle += collisions_curr_tick

        self.prev_drone_collisions = self.curr_drone_collisions

    def calculate_reward(self):
        # Penalties for collisions
        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # Penalties for being too close to other drones
        rew_proximity = -1.0 * calculate_drone_proximity_penalties(
            distance_matrix=self.distance_matrix, dt=self.control_dt, penalty_fall_off=self.collision_falloff_radius,
            max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents)

        return rew_collisions, rew_proximity

    def down_wash(self, real_positions, real_rotations):
        # Apply physical interactions; Change vel & omega

        if self.use_downwash:
            downwash_velocities_change, downwash_omegas_change = perform_downwash(
                num_agents=self.num_agents, positions=real_positions, rotations=real_rotations, dt=self.control_dt)
        else:
            downwash_velocities_change = np.zeros((self.num_agents, 3))
            downwash_omegas_change = np.zeros((self.num_agents, 3))

        return downwash_velocities_change, downwash_omegas_change

    def perform_physical_interaction(self, real_positions, real_velocities, real_rotations):
        downwash_velocities_change, downwash_omegas_change = \
            self.down_wash(real_positions=real_positions, real_rotations=real_rotations)

        neighbor_velocities_change, neighbor_omegas_change = get_vel_omega_change_neighbor_collisions(
            num_agents=self.num_agents, curr_drone_collisions=self.curr_drone_collisions, real_positions=real_positions,
            real_velocities=real_velocities, col_coeff=self.rew_coeff["quadcol_coeff"])

        velocities_change = downwash_velocities_change + neighbor_velocities_change
        omegas_change = downwash_omegas_change + neighbor_omegas_change

        return velocities_change, omegas_change
