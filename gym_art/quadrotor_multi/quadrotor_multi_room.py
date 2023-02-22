import copy

import numpy as np

from gym_art.quadrotor_multi.utils.quad_room_utils import perform_collision_with_wall, perform_collision_with_ceiling


class Room:
    def __init__(self, num_agents, rew_coeff):
        # Pre-requisite
        self.num_agents = num_agents
        self.rew_coeff = rew_coeff

        # Collisions number
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_walls_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        # Collisions list
        self.prev_collisions_room = []
        self.prev_collisions_floor = []
        self.prev_collisions_walls = []
        self.prev_collisions_ceiling = []

        # Reward
        self.emergent_collisions_floor = np.array([])
        self.emergent_collisions_walls = np.array([])
        self.emergent_collisions_ceiling = np.array([])

        # Physical interaction
        self.wall_crash_list = []
        self.ceiling_crash_list = []

    def reset(self, rew_coeff):
        # Update
        self.rew_coeff = rew_coeff

        # Collisions number
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_walls_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        # Collisions list
        self.prev_collisions_room = []
        self.prev_collisions_floor = []
        self.prev_collisions_walls = []
        self.prev_collisions_ceiling = []

    def step(self):
        pass

    def get_collision_reward(self, emergent_collisions, rew_coeff=0.0):
        rew_raw = np.zeros(self.num_agents)
        if emergent_collisions.any():
            rew_raw[emergent_collisions] = -1.0
        rew = rew_coeff * rew_raw

        return rew

    def calculate_reward(self):
        rew_floor = self.get_collision_reward(emergent_collisions=self.emergent_collisions_floor,
                                              rew_coeff=self.rew_coeff["crash_room"])

        rew_walls = self.get_collision_reward(emergent_collisions=self.emergent_collisions_walls,
                                              rew_coeff=self.rew_coeff["crash_room"])

        rew_ceiling = self.get_collision_reward(emergent_collisions=self.emergent_collisions_ceiling,
                                                rew_coeff=self.rew_coeff["crash_room"])

        return rew_floor, rew_walls, rew_ceiling

    def perform_physical_interaction(self, real_positions, real_velocities, real_omegas, room_box):
        new_velocities = copy.deepcopy(real_velocities)
        new_omegas = copy.deepcopy(real_omegas)

        for val in self.wall_crash_list:
            drone_id = int(val)
            dyn_vel, dyn_omega = perform_collision_with_wall(
                dyn_pos=real_positions[drone_id], dyn_vel=real_velocities[drone_id], room_box=room_box)

            new_velocities[drone_id] = dyn_vel
            new_omegas[drone_id] = dyn_omega

        for val in self.ceiling_crash_list:
            drone_id = int(val)
            dyn_vel, dyn_omega = perform_collision_with_ceiling(dyn_vel=real_velocities[drone_id])

            new_velocities[drone_id] = dyn_vel
            new_omegas[drone_id] = dyn_omega

        return new_velocities, new_omegas

    def calculate_collision_info(self, envs, rew_coeff):
        # Update
        self.rew_coeff = rew_coeff

        apply_room_collision_flag = False
        floor_collisions = np.array([env.dynamics.crashed_floor for env in envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        self.wall_crash_list = np.where(wall_collisions >= 1)[0]
        if len(self.wall_crash_list) > 0:
            apply_room_collision_flag = True

        self.ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        if len(self.ceiling_crash_list) > 0:
            apply_room_collision_flag = True

        room_crash_list = np.unique(np.concatenate([floor_crash_list, self.wall_crash_list, self.ceiling_crash_list]))

        # Get emergent collisions
        emergent_collisions_room = np.setdiff1d(room_crash_list, self.prev_collisions_room)
        emergent_collisions_floor = np.setdiff1d(floor_crash_list, self.prev_collisions_floor)
        emergent_collisions_walls = np.setdiff1d(self.wall_crash_list, self.prev_collisions_walls)
        emergent_collisions_ceiling = np.setdiff1d(self.ceiling_crash_list, self.prev_collisions_ceiling)

        # Calculate total collisions per episode for room, floor, walls, ceiling
        self.collisions_room_per_episode += len(emergent_collisions_room)
        self.collisions_floor_per_episode += len(emergent_collisions_floor)
        self.collisions_walls_per_episode += len(emergent_collisions_walls)
        self.collisions_ceiling_per_episode += len(emergent_collisions_ceiling)

        # Set crash list to as previous crash list
        self.prev_collisions_room = room_crash_list
        self.prev_collisions_floor = floor_crash_list
        self.prev_collisions_walls = self.wall_crash_list
        self.prev_collisions_ceiling = self.ceiling_crash_list

        return apply_room_collision_flag
