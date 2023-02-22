import numpy as np

from gym_art.quadrotor_multi.utils.quad_utils import compute_new_vel, compute_new_omega, EPS, QUAD_RADIUS


# Reward: dt * penalties
def calculate_obst_drone_proximity_penalties(distances, dt, penalty_fall_off, max_penalty, num_agents):
    if not penalty_fall_off:
        # smooth penalties is disabled
        return np.zeros(num_agents)

    dist_ratio = 1 - distances / (penalty_fall_off * QUAD_RADIUS)
    dist_ratio = np.maximum(dist_ratio, 0)
    penalties = dist_ratio * max_penalty

    return dt * penalties


def compute_col_norm_and_new_vel_obst(dyn_pos, dyn_vel, obstacle_pos):
    collision_norm = dyn_pos - obstacle_pos
    # difference in z position is 0, given obstacle height is same as room height
    collision_norm[2] = 0.0
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    vnew = np.dot(dyn_vel, collision_norm)

    return vnew, collision_norm


# Collision model
def perform_collision_with_obstacle(dyn_pos, dyn_vel, obstacle_pos, obstacle_size):
    # Vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    vnew, collision_norm = compute_col_norm_and_new_vel_obst(dyn_pos=dyn_pos, dyn_vel=dyn_vel,
                                                             obstacle_pos=obstacle_pos)
    vel_change = -vnew * collision_norm

    dyn_vel_shift = vel_change
    for _ in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        vel_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        dyn_vel_shift = vel_change + vel_noise
        if np.dot(dyn_vel + dyn_vel_shift, collision_norm) > 0:
            break

    max_vel_magn = np.linalg.norm(dyn_vel)
    if np.linalg.norm(dyn_pos - obstacle_pos) <= obstacle_size / 2.0:
        dyn_vel_change = compute_new_vel(max_vel_magn=max_vel_magn, vel=dyn_vel, vel_change=dyn_vel_shift, low=1.0,
                                         high=1.0)
    else:
        dyn_vel_change = compute_new_vel(max_vel_magn=max_vel_magn, vel=dyn_vel, vel_change=dyn_vel_shift)

    # Random forces for omega
    new_omega = compute_new_omega()
    dyn_omega_change = new_omega

    return dyn_vel_change, dyn_omega_change


def get_vel_omega_change_obst_collisions(num_agents, obst_quad_col_matrix, real_positions, real_velocities,
                                         obstacle_size, obstacle_poses, col_coeff):
    velocities_change = np.zeros((num_agents, 3))
    omegas_change = np.zeros((num_agents, 3))
    for i, val in enumerate(obst_quad_col_matrix):
        drone_id = int(val)
        dyn_vel_change, dyn_omega_change = perform_collision_with_obstacle(
            dyn_pos=real_positions[drone_id], dyn_vel=real_velocities[drone_id], obstacle_pos=obstacle_poses[i],
            obstacle_size=obstacle_size)

        velocities_change[drone_id] += dyn_vel_change
        omegas_change[drone_id] += dyn_omega_change

    return velocities_change * col_coeff, omegas_change * col_coeff
