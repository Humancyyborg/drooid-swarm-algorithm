import numpy as np

from gym_art.quadrotor_multi.utils.quad_utils import compute_new_vel, compute_new_omega, EPS


# Reward: dt * penalties
def calculate_obst_drone_proximity_penalties(distances, arm, dt, penalty_fall_off, max_penalty, num_agents):
    if not penalty_fall_off:
        # smooth penalties is disabled
        return np.zeros(num_agents)

    dist_ratio = 1 - distances / (penalty_fall_off * arm)
    dist_ratio = np.maximum(dist_ratio, 0)
    penalties = dist_ratio * max_penalty

    return dt * penalties


def compute_col_norm_and_new_vel_obst(dyn, obstacle_pos):
    collision_norm = dyn.pos - obstacle_pos
    # difference in z position is 0, given obstacle height is same as room height
    collision_norm[2] = 0.0
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    vnew = np.dot(dyn.vel, collision_norm)

    return vnew, collision_norm


# Collision model
def perform_collision_with_obstacle(drone_dyn, obstacle_pos, obstacle_size, col_coeff=1.0):
    # Vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    vnew, collision_norm = compute_col_norm_and_new_vel_obst(drone_dyn, obstacle_pos)
    vel_change = -vnew * collision_norm

    dyn_vel_shift = vel_change
    for _ in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        vel_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        dyn_vel_shift = vel_change + vel_noise
        if np.dot(drone_dyn.vel + dyn_vel_shift, collision_norm) > 0:
            break

    max_vel_magn = np.linalg.norm(drone_dyn.vel)
    if np.linalg.norm(drone_dyn.pos - obstacle_pos) <= obstacle_size:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel, vel_shift=dyn_vel_shift,
                                        coeff=col_coeff, low=1.0, high=1.0)
    else:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel, vel_shift=dyn_vel_shift,
                                        coeff=col_coeff)

    # Random forces for omega
    new_omega = compute_new_omega()
    drone_dyn.omega += new_omega * col_coeff
