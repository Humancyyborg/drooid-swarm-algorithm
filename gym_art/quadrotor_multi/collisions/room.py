import numba as nb
import numpy as np
from numba import njit


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


@njit
def perform_collision_with_wall_numba(vel, pos, omega, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                      lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # Decrease drone's speed after collision with wall
    drone_speed = np.linalg.norm(vel)
    real_speed = nb.random.uniform(damp_low_speed_ratio * drone_speed, damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, 0.1, 6.0)

    drone_pos = pos
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    direction = np.random.uniform(-1.0, 1.0, size=(3,))
    if x_list[0]:
        direction[0] = np.random.uniform(0.1, 1.0)
    elif x_list[1]:
        direction[0] = np.random.uniform(-1.0, -0.1)

    if y_list[0]:
        direction[1] = np.random.uniform(0.1, 1.0)
    elif y_list[1]:
        direction[1] = np.random.uniform(-1.0, -0.1)

    direction[2] = np.random.uniform(-1.0, -0.5)

    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    vel = real_speed * direction_norm

    # Random forces for omega
    omega_max = 20 * np.pi  # this will amount to max 3.5 revolutions per second
    new_omega = np.random.uniform(-1, 1, size=(3,))  # random direction in 3D space
    new_omega /= np.linalg.norm(new_omega) + eps  # normalize

    new_omega_mag = np.random.uniform(omega_max / 2, omega_max)  # random magnitude of the force
    new_omega *= new_omega_mag

    # add the disturbance to drone's angular velocities while preserving angular momentum
    omega += new_omega

    return vel, omega


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
