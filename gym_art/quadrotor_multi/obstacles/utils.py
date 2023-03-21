import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import QUAD_RADIUS


@njit
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos)
                    if dist < min_dist:
                        min_dist = dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist - obst_radius

    return quads_sdf_obs


@njit
def collision_detection(quad_poses, obst_poses, obst_radius):
    quad_num = len(quad_poses)
    collide_threshold = QUAD_RADIUS + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    cell_centers = [
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in
        np.arange(0, obst_area_length, grid_size) for j in
        np.arange(obst_area_width - grid_size, -grid_size, -grid_size)]
    return cell_centers


if __name__ == "__main__":
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    quads_sdf_obs = 100 * np.ones((len(quad_poses), 9))

    dist = []
    for i, x in enumerate([-0.1, 0, 0.1]):
        for j, y in enumerate([-0.1, 0, 0.1]):
            tmp = np.linalg.norm([x - obst_poses[0][0], y - obst_poses[0][1]]) - 0.3
            dist.append(tmp)

    test_res = get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius=0.3, resolution=0.1)
    true_res = np.array(dist)
    # print('algo:    ', test_res)
    # print('true:    ', true_res)
    assert test_res.all() == true_res.all()

    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3)
    test_res = np.where(quad_collisions > -1)[0]
    true_res = -1 * np.ones(len(quad_poses), dtype=int)
    if dist[4] < 0:
        true_res[0] = int(0)

    # print('algo:    ', test_res)
    # print('true:    ', true_res)

    assert test_res.all() == true_res.all()

    print('Pass unit test!')
