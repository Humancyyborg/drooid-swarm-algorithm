import numpy as np

from gym_art.quadrotor_multi.utils.quad_utils import EPS, compute_new_omega


def perform_collision_with_wall(dyn_pos, dyn_vel, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                lowest_speed=0.1, highest_speed=6.0):
    # Decrease drone's speed after collision with wall
    drone_speed = np.linalg.norm(dyn_vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    x_list = [dyn_pos[0] == room_box[0][0], dyn_pos[0] == room_box[1][0]]
    y_list = [dyn_pos[1] == room_box[0][1], dyn_pos[1] == room_box[1][1]]

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
    direction_norm = direction / (direction_mag + EPS)

    dyn_vel = real_speed * direction_norm
    dyn_omega = compute_new_omega()

    # We directly change velocity and omega
    return dyn_vel, dyn_omega


def perform_collision_with_ceiling(dyn_vel, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                   lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    drone_speed = np.linalg.norm(dyn_vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    direction[2] = np.random.uniform(low=-1.0, high=-0.5)
    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    dyn_vel = real_speed * direction_norm
    dyn_omega = compute_new_omega()

    return dyn_vel, dyn_omega
