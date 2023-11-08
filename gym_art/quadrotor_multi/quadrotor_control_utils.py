import numpy as np
from numba import njit

GRAV = 9.81


def _compute_maximum_distance_to_boundary(acc, G, h):
    """
        Computes the maximum distance of v to the constraint boundary
        defined by Gx <= h.

        If a constraint is satisfied by v, the distance of v to the
        constraint boundary is negative. Constraints take value zero at the
        boundary. Distance to violated constraints is positive.

        If the return value is positive, the return value is the distance
        to the most violated constraint.

        If it is non-positive, all constraints are satisfied,
        and the return value is the (negated) distance to the
        closest constraint.

        In any case, we want this value to be minimized.
    """
    G_norms = np.linalg.norm(G, axis=1)
    G_unit = G / G_norms[:, np.newaxis]
    h_norm = h / G_norms

    # Compute the distance of the acceleration to the constraint boundaries
    distances = np.dot(G_unit, acc) - h_norm
    return np.maximum(distances.max(), 0)


@njit
def get_min_real_dist(min_rel_dist, self_state_pos, description_state_pos, rel_pos_arr, rel_pos_norm_arr,
                      neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_pos = self_state_pos - description_state_pos[idx]
        rel_pos_norm = np.linalg.norm(rel_pos)

        rel_pos_arr[idx] = rel_pos
        rel_pos_norm_arr[idx] = rel_pos_norm
        min_rel_dist = min(rel_pos_norm, min_rel_dist)

    return min_rel_dist


@njit
def get_obst_min_real_dist(min_rel_dist, self_state_pos, description_state_pos, rel_pos_arr, rel_pos_norm_arr,
                           neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_pos = self_state_pos - description_state_pos[idx]
        rel_pos_norm = np.linalg.norm(rel_pos)

        rel_pos_arr[idx] = rel_pos
        rel_pos_norm_arr[idx] = rel_pos_norm
        min_rel_dist = min(rel_pos_norm, min_rel_dist)

    return min_rel_dist


@njit
def get_G_h(self_state_vel, neighbor_descriptions, rel_pos_arr, rel_pos_norm_arr, safety_distance,
            maximum_linf_acceleration, aggressiveness, G, h, start_id, max_lin_acc, neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_vel = self_state_vel - neighbor_descriptions[idx]
        rel_pos_dot_rel_vel = np.dot(rel_pos_arr[idx], rel_vel)
        safe_rel_dist = rel_pos_norm_arr[idx] - safety_distance
        react_time = rel_pos_dot_rel_vel / rel_pos_norm_arr[idx]

        hij = np.sqrt(2.0 * max_lin_acc * safe_rel_dist) + react_time

        rel_pos_dot_vel = np.dot(rel_pos_arr[idx], self_state_vel)
        rel_vel_dot_vel = np.dot(rel_vel, self_state_vel)

        # item_chunk_prev
        item_chunk_prev = -rel_pos_dot_rel_vel * rel_pos_dot_vel / (rel_pos_norm_arr[idx] ** 2)

        # item_chunk_post
        chunk_coeff = maximum_linf_acceleration / max_lin_acc
        chunk_rel_pos = aggressiveness * (hij ** 3) * rel_pos_norm_arr[idx]
        chunk_rel_pos_dot_rel_vel = np.sqrt(max_lin_acc) * rel_pos_dot_rel_vel
        chunk_safe_rel_dist = np.sqrt(2.0 * safe_rel_dist)
        chunk_value = chunk_rel_pos + chunk_rel_pos_dot_rel_vel / chunk_safe_rel_dist
        item_chunk_post = chunk_coeff * chunk_value

        bij = item_chunk_prev + rel_vel_dot_vel + item_chunk_post

        G[start_id + idx] = -1.0 * rel_pos_arr[idx]
        h[start_id + idx] = bij

    return G, h


@njit
def get_obst_G_h(self_state_vel, neighbor_descriptions, rel_pos_arr, rel_pos_norm_arr, safety_distance,
                 maximum_linf_acceleration, aggressiveness, G, h, start_id, max_lin_acc, neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_vel = self_state_vel - neighbor_descriptions[idx]
        rel_pos_dot_rel_vel = np.dot(rel_pos_arr[idx], rel_vel)
        safe_rel_dist = rel_pos_norm_arr[idx] - safety_distance
        react_time = rel_pos_dot_rel_vel / rel_pos_norm_arr[idx]

        hij = np.sqrt(2.0 * max_lin_acc * safe_rel_dist) + react_time

        rel_pos_dot_vel = np.dot(rel_pos_arr[idx], self_state_vel)
        rel_vel_dot_vel = np.dot(rel_vel, self_state_vel)

        # item_chunk_prev
        item_chunk_prev = -rel_pos_dot_rel_vel * rel_pos_dot_vel / (rel_pos_norm_arr[idx] ** 2)

        # item_chunk_post
        chunk_coeff = maximum_linf_acceleration / max_lin_acc
        chunk_rel_pos = aggressiveness * (hij ** 3) * rel_pos_norm_arr[idx]
        chunk_rel_pos_dot_rel_vel = np.sqrt(max_lin_acc) * rel_pos_dot_rel_vel
        chunk_safe_rel_dist = np.sqrt(2.0 * safe_rel_dist)
        chunk_value = chunk_rel_pos + chunk_rel_pos_dot_rel_vel / chunk_safe_rel_dist
        item_chunk_post = chunk_coeff * chunk_value

        bij = item_chunk_prev + rel_vel_dot_vel + item_chunk_post

        tmp = -1.0 * rel_pos_arr[idx]
        G[start_id + idx] = np.array([tmp[0], tmp[1], 0.0])
        h[start_id + idx] = bij

    return G, h


@njit
def get_G_h_room(self_state_vel, neighbor_descriptions, rel_pos_arr, rel_pos_norm_arr, safety_distance,
                 maximum_linf_acceleration, aggressiveness, G, h, start_id):
    max_lin_acc = maximum_linf_acceleration
    for idx in range(len(neighbor_descriptions)):
        rel_vel = self_state_vel[int(idx // 2)]
        d_self_state_vel = self_state_vel[int(idx // 2)]
        rel_pos_dot_rel_vel = rel_pos_arr[idx] * rel_vel
        safe_rel_dist = rel_pos_norm_arr[idx] - safety_distance
        react_time = rel_pos_dot_rel_vel / rel_pos_norm_arr[idx]

        hij = np.sqrt(2.0 * max_lin_acc * safe_rel_dist) + react_time

        rel_pos_dot_vel = rel_pos_arr[idx] * d_self_state_vel
        rel_vel_dot_vel = rel_vel * d_self_state_vel

        # item_chunk_prev
        item_chunk_prev = -rel_pos_dot_rel_vel * rel_pos_dot_vel / (rel_pos_norm_arr[idx] ** 2)

        # item_chunk_post
        chunk_coeff = 1.0
        chunk_rel_pos = aggressiveness * (hij ** 3) * rel_pos_norm_arr[idx]
        chunk_rel_pos_dot_rel_vel = np.sqrt(max_lin_acc) * rel_pos_dot_rel_vel
        chunk_safe_rel_dist = np.sqrt(2.0 * safe_rel_dist)
        chunk_value = chunk_rel_pos + chunk_rel_pos_dot_rel_vel / chunk_safe_rel_dist
        item_chunk_post = chunk_coeff * chunk_value

        bij = item_chunk_prev + rel_vel_dot_vel + item_chunk_post

        coefficients = np.zeros(3)
        coefficients[int(idx // 2)] = -1.0 * rel_pos_arr[idx]
        G[start_id + idx] = coefficients
        h[start_id + idx] = bij

    return G, h
