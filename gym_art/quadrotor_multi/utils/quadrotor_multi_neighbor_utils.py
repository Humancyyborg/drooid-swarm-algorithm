import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.utils.quad_utils import QUAD_RADIUS

# Count collisions after 1.5 seconds
COL_GRACE_PERIOD = 1.5


def calculate_neighbor_collision_matrix(positions, hit_box_size):
    dist_matrix = spatial.distance_matrix(x=positions, y=positions)
    collision_matrix = (dist_matrix < hit_box_size * QUAD_RADIUS).astype(np.float32)
    np.fill_diagonal(collision_matrix, 0.0)

    # get upper triangular matrix and check if they have collisions and append to all collisions
    upt = np.triu(collision_matrix)
    up_w1 = np.where(upt >= 1)
    collisions_pair_list = []
    for i, val in enumerate(up_w1[0]):
        collisions_pair_list.append((up_w1[0][i], up_w1[1][i]))

    return collision_matrix, collisions_pair_list, dist_matrix


def calculate_proximity_rewards(dist_matrix, dt, prox_box_size, max_penalty, num_agents):
    if not prox_box_size:
        # smooth penalties is disabled
        return np.zeros(num_agents)
    penalties = (-max_penalty / (prox_box_size * QUAD_RADIUS)) * dist_matrix + max_penalty
    np.fill_diagonal(penalties, 0.0)
    penalties = np.maximum(penalties, 0.0)
    penalties = np.sum(penalties, axis=0)

    # Actual penalties per tick to be added to the overall reward
    return dt * penalties


def set_neighbor_interaction(num_agents, pos, hit_box_size, prox_box_size, cur_tick, control_freq,
                             prev_collisions, collisions_per_episode, collisions_after_settle,
                             rew_coeff_neighbor, rew_coeff_neighbor_prox):
    # Calculating collisions between drones
    collision_matrix, collisions_pair_list, dist_matrix = calculate_neighbor_collision_matrix(positions=pos,
                                                                                              hit_box_size=hit_box_size)

    emergent_collisions = np.setdiff1d(collisions_pair_list, prev_collisions)
    prev_collisions = collisions_pair_list

    # collision between 2 drones counts as a single collision
    collisions_curr_tick = len(emergent_collisions) // 2
    collisions_per_episode += collisions_curr_tick
    if collisions_curr_tick > 0 and cur_tick >= COL_GRACE_PERIOD * control_freq:
        collisions_after_settle += collisions_curr_tick

    rew_collisions_raw = np.zeros(num_agents)
    if emergent_collisions.any():
        rew_collisions_raw[emergent_collisions] = -1.0
    rew_collisions = rew_coeff_neighbor * rew_collisions_raw

    # penalties for being too close to other drones
    rew_collision_proximity = -1.0 * calculate_proximity_rewards(
        dist_matrix=dist_matrix, dt=1.0 / control_freq,
        prox_box_size=prox_box_size,
        max_penalty=rew_coeff_neighbor_prox,
        num_agents=num_agents,
    )

    return prev_collisions, collisions_per_episode, collisions_after_settle, rew_collisions, rew_collision_proximity, \
        collision_matrix, collisions_pair_list, emergent_collisions
