import timeit
import numpy as np


def test_speed_get_cell_centers():
    SETUP_CODE = '''from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers'''

    TEST_CODE = '''get_cell_centers(obst_area_length=8.0, obst_area_width=8.0, grid_size=1.0)'''

    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e4))

    # print('times:   ', times)
    print('get_cell_centers - Mean Time:   ', np.mean(times[1:]))


def test_speed_get_surround_sdfs_radar_2d():
    SETUP_CODE = '''from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs_radar_2d;import numpy as np'''

    TEST_CODE = '''quad_poses=np.array([[0., 0.], [0.5, 0.5], [-0.5, 0.5], [0.8, 0.5], [-0.5, 0.5], [-0.1, 0.9], [1.2, 0.8], [2.5, 1.5]]); obst_poses=np.array([[0., 0.95], [1.5, 2.5], [-0.5, 1.5], [-2.5, 0.5], [-3.5, 1.5], [0.0, 2.5], [1.5, 1.5], [2.5, -1.5]]); quad_vels=np.array([[0.11, 0.25], [-0.5, 0.11], [-1.5, -0.68], [0.77, 0.0], [0.0, 0.0], [0.0, 0.89], [0.25, 0.001], [2.5, -2.2]]);obst_radius=0.3;scan_range=1.5*np.pi;ray_num=8;room_dims=np.array([10., 10., 10.]);get_surround_sdfs_radar_2d(quad_poses, obst_poses, quad_vels, obst_radius, scan_range, ray_num, room_dims)'''

    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e4))

    print('get_surround_sdfs_radar_2d - Mean Time:   ', np.mean(times[1:]))


def speed_test():
    test_speed_get_cell_centers()
    test_speed_get_surround_sdfs_radar_2d()
    return


if __name__ == "__main__":
    speed_test()
