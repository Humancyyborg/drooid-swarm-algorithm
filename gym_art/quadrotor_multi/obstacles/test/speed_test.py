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


def speed_test():
    test_speed_get_cell_centers()
    return


if __name__ == "__main__":
    test_speed_get_cell_centers()
