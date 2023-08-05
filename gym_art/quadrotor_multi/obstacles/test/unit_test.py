import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_cell_centers, \
    get_surround_sdfs_radar_2d


def test_get_surround_sdfs():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    quads_sdf_obs = 100 * np.ones((len(quad_poses), 9))

    # get_surround_sdfs
    dist = []
    for i, x in enumerate([-0.1, 0, 0.1]):
        for j, y in enumerate([-0.1, 0, 0.1]):
            tmp = np.linalg.norm([x - obst_poses[0][0], y - obst_poses[0][1]]) - 0.3
            dist.append(tmp)

    test_res = get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius=0.3, resolution=0.1)
    true_res = np.array(dist)
    assert np.array_equiv(np.round(test_res, decimals=5), np.round(true_res, decimals=5))
    return


def test_collision_detection():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    # collision_detection
    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3, quad_radius=0.046)
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    assert np.array_equiv(test_res, true_res)
    return


def test_get_cell_centers():
    obst_area_length = 8.0
    obst_area_width = 8.0
    grid_size = 1.0
    test_res = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width, grid_size=grid_size)

    true_res = np.array([
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in np.arange(0, obst_area_length, grid_size)
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size)])

    assert np.array_equiv(test_res, true_res)
    return


def test_get_surround_sdfs_radar_2d():
    quad_poses = np.array([[0., 0.]])
    quad_vels = np.array([[0., 1.]])
    obst_poses = np.array([[0., 0.5]])
    room_dims = np.array([10., 10., 10.])
    true_res = np.array([[5., 0.2671814623744493, 0.2671814623744493, 5.]])
    test_res = get_surround_sdfs_radar_2d(quad_poses=quad_poses, obst_poses=obst_poses, quad_vels=quad_vels,
                                          obst_radius=0.3, scan_range=np.pi, ray_num=4, room_dims=room_dims)
    # print('test_res:     ', test_res)
    # print('true_res:     ', true_res)
    assert np.array_equiv(np.round(test_res, decimals=5), np.round(true_res, decimals=5))

    true_res = np.array([[5., 5.0 * np.sqrt(2), 0.2, 5.0 * np.sqrt(2), 5.]])
    test_res = get_surround_sdfs_radar_2d(quad_poses=quad_poses, obst_poses=obst_poses, quad_vels=quad_vels,
                                          obst_radius=0.3, scan_range=np.pi, ray_num=5, room_dims=room_dims)
    # print('test_res:     ', test_res)
    # print('true_res:     ', true_res)
    assert np.array_equiv(np.round(test_res, decimals=5), np.round(true_res, decimals=5))

    quad_vels = np.array([[1., 0.]])
    true_res = np.array([[0.2, 5.0 * np.sqrt(2), 5.0, 5.0 * np.sqrt(2), 5.0]])
    test_res = get_surround_sdfs_radar_2d(quad_poses=quad_poses, obst_poses=obst_poses, quad_vels=quad_vels,
                                          obst_radius=0.3, scan_range=np.pi, ray_num=5, room_dims=room_dims)
    # print('test_res:     ', test_res)
    # print('true_res:     ', true_res)
    assert np.array_equiv(np.round(test_res, decimals=5), np.round(true_res, decimals=5))

    quad_poses = np.array([[0., 0.], [0.5, 0.5]])
    quad_vels = np.array([[0., 1.], [1., 0.]])
    obst_poses = np.array([[0., 0.5], [0.5, 0.]])
    true_res = np.array([[5., 5.0 * np.sqrt(2), 0.2, 5.0 * np.sqrt(2), 0.2],
                         [4.5, 4.5*np.sqrt(2), 4.5, 4.5*np.sqrt(2), 0.2]])
    test_res = get_surround_sdfs_radar_2d(quad_poses=quad_poses, obst_poses=obst_poses, quad_vels=quad_vels,
                                          obst_radius=0.3, scan_range=np.pi, ray_num=5, room_dims=room_dims)
    # print('test_res:     ', test_res)
    # print('true_res:     ', true_res)
    assert np.array_equiv(np.round(test_res, decimals=5), np.round(true_res, decimals=5))


def unit_test():
    test_get_surround_sdfs()
    test_collision_detection()
    test_get_cell_centers()
    test_get_surround_sdfs_radar_2d()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
