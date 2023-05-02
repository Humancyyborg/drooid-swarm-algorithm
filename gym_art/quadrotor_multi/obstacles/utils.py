import numpy as np
from numba import njit


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
def get_surround_sdfs_radar_2d(quad_poses, obst_poses, quad_vels, obst_radius, scan_range, ray_num,
                               room_dims, scan_max_dist):
    """
        quad_poses:     quadrotor positions, only with xy pos
        obst_poses:     obstacle positions, only with xy pos
        quad_vels:      quadrotor velocities, only with xy vel
        obst_radius:    obstacle raidus
        scan_range:     scan range, in radians [pi / 2, pi]
        ray_num:        ray number
    """
    quads_sdf_obs = scan_max_dist * np.ones((len(quad_poses), ray_num))
    scan_angle = scan_range / (ray_num - 1)
    scan_angle_arr = np.zeros(ray_num)
    # room_x = room_dims[0] / 2
    # room_y = room_dims[1] / 2
    # Get ray angle list
    # If scan_range = 180 deg and ray_num = 4
    # scan_angle_arr = [90, 30, -30, -90]
    start_angle = scan_range / 2.
    for rid in range(ray_num):
        scan_angle_arr[rid] = start_angle
        start_angle -= scan_angle

    for q_id in range(len(quad_poses)):
        q_pos_xy = quad_poses[q_id]
        q_vel_xy = quad_vels[q_id]
        base_rad = np.arctan2(q_vel_xy[1], q_vel_xy[0])
        for ray_id, rad_shift in enumerate(scan_angle_arr):
            cur_rad = base_rad + rad_shift
            cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
            for o_pos_xy in obst_poses:
                # Check if the obstacle intersect with the quadrotor
                rel_obst_quad_xy = o_pos_xy - q_pos_xy
                rel_obst_quad_xy_mag = (rel_obst_quad_xy[0] ** 2 + rel_obst_quad_xy[1] ** 2) ** 0.5

                rel_dot_obst_quad_xy = np.dot(cur_dir, rel_obst_quad_xy)
                cos_obst_ray_rad = rel_dot_obst_quad_xy / rel_obst_quad_xy_mag
                if cos_obst_ray_rad == 1.0:
                    quads_sdf_obs[q_id][ray_id] = min(quads_sdf_obs[q_id][ray_id], rel_dot_obst_quad_xy - obst_radius)
                else:
                    if cos_obst_ray_rad <= 0.0:
                        if rel_obst_quad_xy_mag <= obst_radius:
                            quads_sdf_obs[q_id][ray_id] = 0.0
                    else:
                        obst_ray_rad = np.arccos(cos_obst_ray_rad)
                        closest_dist = rel_obst_quad_xy_mag * np.sin(obst_ray_rad)
                        if closest_dist <= obst_radius:
                            len_in_obst = (obst_radius ** 2 - closest_dist ** 2) ** 0.5
                            quads_sdf_obs[q_id][ray_id] = min(quads_sdf_obs[q_id][ray_id],
                                                              rel_dot_obst_quad_xy - len_in_obst)

        # for quad_sdf_obs in quads_sdf_obs:
        # for q_id in range(len(quad_poses)):
        #     q_pos_xy = quad_poses[q_id]
        #     q_vel_xy = quad_vels[q_id]
        #     base_rad = np.arctan2(q_vel_xy[1], q_vel_xy[0])
        #     for ray_id, rad_shift in enumerate(scan_angle_arr):
        #         if quads_sdf_obs[q_id][ray_id] >= 100.0:
        #             cur_rad = base_rad + rad_shift
        #             cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])
        #             # Wall: from [-5., 0.] -> [5., 0.]
        #             for wall_id in range(4):
        #                 if wall_id == 0:
        #                     rel_obst_quad_xy = np.array([-room_x - q_pos_xy[0], 0.0])
        #                 elif wall_id == 1:
        #                     rel_obst_quad_xy = np.array([0.0, room_y - q_pos_xy[1]])
        #                 elif wall_id == 2:
        #                     rel_obst_quad_xy = np.array([room_x - q_pos_xy[0], 0.0])
        #                 else:
        #                     rel_obst_quad_xy = np.array([0.0, -room_y - q_pos_xy[1]])
        #
        #                 rel_dot_obst_quad_xy = np.dot(cur_dir, rel_obst_quad_xy)
        #                 if rel_dot_obst_quad_xy > 0.0:
        #                     rel_obst_quad_xy_mag = (rel_obst_quad_xy[0] ** 2 + rel_obst_quad_xy[1] ** 2) ** 0.5
        #                     cos_obst_ray_rad = rel_dot_obst_quad_xy / rel_obst_quad_xy_mag
        #                     if cos_obst_ray_rad == 1.0:
        #                         quads_sdf_obs[q_id][ray_id] = min(quads_sdf_obs[q_id][ray_id], rel_dot_obst_quad_xy)
        #                     else:
        #                         quads_sdf_obs[q_id][ray_id] = min(quads_sdf_obs[q_id][ray_id],
        #                                                           rel_obst_quad_xy_mag / cos_obst_ray_rad)

    return quads_sdf_obs


@njit
def collision_detection(quad_poses, obst_poses, obst_radius, quad_radius):
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    count = 0
    i_len = obst_area_length / grid_size
    j_len = obst_area_width / grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, obst_area_length, grid_size):
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    unit_test()
    speed_test()
