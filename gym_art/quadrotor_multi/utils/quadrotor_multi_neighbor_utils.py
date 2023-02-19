import numpy as np
from numba import njit
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


@njit
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


if __name__ == "__main__":
    """
        measure time (s)

        set_neighbor_interaction;
        numba: mean: 0.1368, std: 0.00201
        plain: mean: 0.1565, std: 0.00276
    """

    import timeit

    stmt = 'set_neighbor_interaction(num_agents, pos, hit_box_size, prox_box_size, cur_tick, control_freq, ' \
           'prev_collisions, collisions_per_episode, collisions_after_settle, rew_coeff_neighbor, ' \
           'rew_coeff_neighbor_prox)'
    setup = 'from __main__ import set_neighbor_interaction; import numpy as np; ' \
            'num_agents=8; pos=np.random.uniform(low=-5.0, high=5.0, size=(8,3)); hit_box_size=2.; prox_box_size=4.; cur_tick=500; ' \
            'control_freq=100; prev_collisions=[0]; collisions_per_episode=0; collisions_after_settle=0; ' \
            'rew_coeff_neighbor=1.0; rew_coeff_neighbor_prox=2.0'

    use_numba = True
    # Pass the argument 'n=100' to my_function() and time it
    if use_numba:
        repeat = 6
    else:
        repeat = 5

    t = timeit.Timer(stmt=stmt, setup=setup)
    time_taken = t.repeat(repeat=repeat, number=int(2e3))
    if use_numba:
        time_taken = np.array(time_taken[1:])
    else:
        time_taken = np.array(time_taken)

    print('Time taken:', time_taken)
    print('Time Mean:', time_taken.mean())
    print('Time Std:', time_taken.std())
