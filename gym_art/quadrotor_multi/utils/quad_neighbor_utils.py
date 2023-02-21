import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.utils.quad_utils import compute_new_vel, compute_new_omega, EPS, QUAD_RADIUS

# 1.5 seconds
COLLISIONS_GRACE_PERIOD = 1.5


# Rendering & Reward
# Rendering: collision_matrix
# Reward: 1) all_collisions; 2) dist
def calculate_collision_matrix(positions, hitbox_radius):
    dist = spatial.distance_matrix(x=positions, y=positions)
    collision_matrix = (dist < hitbox_radius * QUAD_RADIUS).astype(np.float32)
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


def compute_neighbor_interaction(num_agents, tick, control_freq, positions, rew_coeff_neighbor, rew_coeff_neighbor_prox,
                                 prev_drone_collisions, collisions_per_episode, collisions_after_settle,
                                 collision_hitbox_radius, collision_falloff_radius):

    # Calculating collisions between drones
    drone_col_matrix, curr_drone_collisions, distance_matrix = \
        calculate_collision_matrix(positions=positions, hitbox_radius=collision_hitbox_radius)

    last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, prev_drone_collisions)

    # collision between 2 drones counts as a single collision
    collisions_curr_tick = len(last_step_unique_collisions) // 2
    collisions_per_episode += collisions_curr_tick

    if collisions_curr_tick > 0:
        if tick >= COLLISIONS_GRACE_PERIOD * control_freq:
            collisions_after_settle += collisions_curr_tick

    prev_drone_collisions = curr_drone_collisions

    rew_collisions_raw = np.zeros(num_agents)
    if last_step_unique_collisions.any():
        rew_collisions_raw[last_step_unique_collisions] = -1.0
    rew_collisions = rew_coeff_neighbor * rew_collisions_raw

    # penalties for being too close to other drones
    rew_proximity = -1.0 * calculate_drone_proximity_penalties(
        distance_matrix=distance_matrix, arm=QUAD_RADIUS, dt=1.0 / control_freq,
        penalty_fall_off=collision_falloff_radius,
        max_penalty=rew_coeff_neighbor_prox,
        num_agents=num_agents,
    )

    return curr_drone_collisions, prev_drone_collisions, rew_collisions_raw, rew_collisions, rew_proximity, \
        collisions_per_episode, collisions_after_settle, drone_col_matrix, last_step_unique_collisions


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


# Used in perform_downwash
def get_vel_omega_norm(z_axis):
    # vel_norm
    noise_z_axis = z_axis + np.random.uniform(low=-0.1, high=0.1, size=3)
    noise_z_axis_mag = np.linalg.norm(noise_z_axis)
    noise_z_axis_norm = noise_z_axis / (noise_z_axis_mag + EPS if noise_z_axis_mag == 0.0 else noise_z_axis_mag)
    down_z_axis_norm = -1.0 * noise_z_axis_norm

    # omega norm
    dir_omega = np.random.uniform(low=-1, high=1, size=3)
    dir_omega_mag = np.linalg.norm(dir_omega)
    dir_omega_norm = dir_omega / (dir_omega_mag + EPS if dir_omega_mag == 0.0 else dir_omega_mag)

    return down_z_axis_norm, dir_omega_norm


# Collision Model
def perform_downwash(drones_dyn, dt):
    # based on some data from Neural-Swarm: https://arxiv.org/pdf/2003.02992.pdf, Fig. 3
    # quadrotor weights: 34 grams
    # 0.5 m, force = 4 grams ; 0.4 m, force = 6 grams
    # 0.3 m, force = 8 grams ; 0.2 m, force = 10 grams
    # force function: f(x) = -20x + 14
    # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17, x in [0, 0.7]
    # Use cylinder to simulate the downwash area
    # The downwash area is a cylinder with radius of 2 arm ~ 10 cm and height of 1.0 m
    xy_downwash = 0.1
    z_downwash = 0.7
    # get pos
    dyns_pos = np.array([d.pos for d in drones_dyn])
    # get z_axis
    dyns_z_axis = np.array([d.rot[:, -1] for d in drones_dyn])

    # drone num
    dyns_num = len(drones_dyn)
    # check if neighbors drones are within the downwash areas, if yes, apply downwash
    for i in range(dyns_num):
        z_axis = dyns_z_axis[i]
        neighbor_pos = dyns_pos - dyns_pos[i]
        neighbor_pos_dist = np.linalg.norm(neighbor_pos, axis=1)
        # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17
        # x in [0, 0.7], a(x) in [0.0, 0.41]
        # acc = (1 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.03, high=0.03)
        acc = (6 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.1, high=0.1)
        acc = np.maximum(EPS, acc)

        # omega downwash given neighbor_pos_dist
        # 0.3 * (x - 1)^2 + random(-0.01, 0.01)
        omega_downwash = 0.3 * (neighbor_pos_dist - 1) ** 2 + np.random.uniform(low=-0.01, high=0.01)
        omega_downwash = np.maximum(EPS, omega_downwash)

        rel_dists_z = np.dot(neighbor_pos, z_axis)
        rel_dists_xy = np.sqrt(neighbor_pos_dist ** 2 - rel_dists_z ** 2)

        for j in range(dyns_num):
            if i == j:
                continue

            if -z_downwash < rel_dists_z[j] < 0 and rel_dists_xy[j] < xy_downwash:
                down_z_axis_norm, dir_omega_norm = get_vel_omega_norm(z_axis=z_axis)
                drones_dyn[j].vel += acc[j] * down_z_axis_norm * dt
                drones_dyn[j].omega += omega_downwash[j] * dir_omega_norm * dt
