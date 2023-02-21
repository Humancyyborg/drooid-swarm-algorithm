import numpy as np

from gym_art.quadrotor_multi.utils.quad_rew_info_utils import get_collision_reward_room


def perform_collision_with_wall(drone_dyn, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # Decrease drone's speed after collision with wall
    drone_speed = np.linalg.norm(drone_dyn.vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    drone_pos = drone_dyn.pos
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    if x_list[0]:
        direction[0] = np.random.uniform(low=0.1, high=1.0)
    elif x_list[1]:
        direction[0] = np.random.uniform(low=-1.0, high=-0.1)

    if y_list[0]:
        direction[1] = np.random.uniform(low=0.1, high=1.0)
    elif y_list[1]:
        direction[1] = np.random.uniform(low=-1.0, high=-0.1)

    direction[2] = np.random.uniform(low=-1.0, high=-0.5)

    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_mag = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    drone_dyn.omega += new_omega


def perform_collision_with_ceiling(drone_dyn, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                   lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    drone_speed = np.linalg.norm(drone_dyn.vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    direction[2] = np.random.uniform(low=-1.0, high=-0.5)
    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    new_omega = np.random.uniform(low=-1, high=1, size=(3,))  # random direction in 3D space
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_mag = np.random.uniform(low=omega_max / 2, high=omega_max)  # random magnitude of the force
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    drone_dyn.omega += new_omega


def simulate_collision_with_room(envs):
    apply_room_collision_flag = False
    floor_collisions = np.array([env.dynamics.crashed_floor for env in envs])
    wall_collisions = np.array([env.dynamics.crashed_wall for env in envs])
    ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in envs])

    floor_crash_list = np.where(floor_collisions >= 1)[0]

    wall_crash_list = np.where(wall_collisions >= 1)[0]
    if len(wall_crash_list) > 0:
        apply_room_collision_flag = True
        for val in wall_crash_list:
            perform_collision_with_wall(drone_dyn=envs[val].dynamics, room_box=envs[0].room_box)

    ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
    if len(ceiling_crash_list) > 0:
        apply_room_collision_flag = True
        for val in ceiling_crash_list:
            perform_collision_with_ceiling(drone_dyn=envs[val].dynamics)

    return apply_room_collision_flag, floor_crash_list, wall_crash_list, ceiling_crash_list


def compute_room_interaction(num_agents, use_replay_buffer, activate_replay_buffer, crashes_last_episode,
                             info_rew_crash,
                             envs, rew_obst_quad_collisions_raw, drone_col_matrix, prev_collisions_room,
                             collisions_room_per_episode, prev_collisions_floor, prev_collisions_walls,
                             prev_collisions_ceiling):
    if use_replay_buffer and not activate_replay_buffer:
        crashes_last_episode += info_rew_crash

    # Collisions with ground
    ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in envs]
    obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
    all_collisions = {'drone': np.sum(drone_col_matrix, axis=1), 'ground': ground_collisions, 'obstacle': obst_coll}

    apply_room_collision, floor_crash_list, wall_crash_list, ceiling_crash_list = \
        simulate_collision_with_room(envs=envs)

    room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))

    emergent_collisions_room = np.setdiff1d(room_crash_list, prev_collisions_room)
    collisions_room_per_episode += len(emergent_collisions_room)

    emergent_collisions_floor = np.setdiff1d(floor_crash_list, prev_collisions_floor)
    emergent_collisions_walls = np.setdiff1d(wall_crash_list, prev_collisions_walls)
    emergent_collisions_ceiling = np.setdiff1d(ceiling_crash_list, prev_collisions_ceiling)

    prev_collisions_room = room_crash_list
    prev_collisions_floor = floor_crash_list
    prev_collisions_walls = wall_crash_list
    prev_collisions_ceiling = ceiling_crash_list

    rew_raw_floor, rew_floor, rew_raw_walls, rew_walls, rew_raw_ceiling, rew_ceiling = \
        get_collision_reward_room(num_agents=num_agents, emergent_collisions_floor=emergent_collisions_floor,
                                  emergent_collisions_walls=emergent_collisions_walls,
                                  emergent_collisions_ceiling=emergent_collisions_ceiling, rew_coeff_floor=0.0,
                                  rew_coeff_walls=0.0, rew_coeff_ceiling=0.0)

    return crashes_last_episode, all_collisions, collisions_room_per_episode, rew_raw_floor, rew_floor, rew_raw_walls, \
        rew_walls, rew_raw_ceiling, rew_ceiling, prev_collisions_room, prev_collisions_floor, prev_collisions_walls, \
        prev_collisions_ceiling, apply_room_collision
