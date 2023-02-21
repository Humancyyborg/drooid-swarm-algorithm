import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.utils.quad_utils import compute_new_vel, compute_new_omega


# Rendering & Reward
# Rendering: collision_matrix
# Reward: 1) all_collisions; 2) dist
def calculate_collision_matrix(positions, arm, hitbox_radius):
    dist = spatial.distance_matrix(x=positions, y=positions)
    collision_matrix = (dist < hitbox_radius * arm).astype(np.float32)
    np.fill_diagonal(collision_matrix, 0.0)

    # get upper triangular matrix and check if they have collisions and append to all collisions
    upt = np.triu(collision_matrix)
    up_w1 = np.where(upt >= 1)
    all_collisions = []
    for i, val in enumerate(up_w1[0]):
        all_collisions.append((up_w1[0][i], up_w1[1][i]))

    return collision_matrix, all_collisions, dist


# Reward: dt * penalties
def calculate_drone_proximity_penalties(distance_matrix, arm, dt, penalty_fall_off, max_penalty, num_agents):
    if not penalty_fall_off:
        # smooth penalties is disabled, so noop
        return np.zeros(num_agents)
    penalties = (-max_penalty / (penalty_fall_off * arm)) * distance_matrix + max_penalty
    np.fill_diagonal(penalties, 0.0)
    penalties = np.maximum(penalties, 0.0)
    penalties = np.sum(penalties, axis=0)

    return dt * penalties  # actual penalties per tick to be added to the overall reward


# Collision model
def compute_col_norm_and_new_velocities(dyn1, dyn2):
    # Ge the collision normal, i.e difference in position
    collision_norm = dyn1.pos - dyn2.pos
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + 0.00001 if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    v1new = np.dot(dyn1.vel, collision_norm)
    v2new = np.dot(dyn2.vel, collision_norm)

    return v1new, v2new, collision_norm


# Collision model
def perform_collision_between_drones(dyn1, dyn2, col_coeff=1.0):
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(dyn1, dyn2)
    vel_change = (v2new - v1new) * collision_norm
    dyn1_vel_shift = vel_change
    dyn2_vel_shift = -vel_change

    # Make sure new vel direction would be opposite to the original vel direction
    for _ in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        vel1_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        vel2_noise = -cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)

        dyn1_vel_shift = vel_change + vel1_noise
        dyn2_vel_shift = -vel_change + vel2_noise

        dyn1_new_vel_dir = np.dot(dyn1.vel + dyn1_vel_shift, collision_norm)
        dyn2_new_vel_dir = np.dot(dyn2.vel + dyn2_vel_shift, collision_norm)

        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            break

    # Get new vel
    max_vel_magn = max(np.linalg.norm(dyn1.vel), np.linalg.norm(dyn2.vel))
    dyn1.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=dyn1.vel, vel_shift=dyn1_vel_shift, coeff=col_coeff)
    dyn2.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=dyn2.vel, vel_shift=dyn2_vel_shift, coeff=col_coeff)

    # Get new omega
    new_omega = compute_new_omega()
    dyn1.omega += new_omega * col_coeff
    dyn2.omega -= new_omega * col_coeff
