import numpy as np

from gym_art.quadrotor_multi.collisions.obstacles import compute_col_norm_and_new_vel_obst


def test_compute_col_norm_and_new_vel_obst():
    quad_pos = np.array([0., 0., 0.])
    quad_vel = np.array([1., 0., 0.])

    obst_pos = np.array([0.5, 0.5, 5.])

    true_vnew = -np.sqrt(2) / 2.
    true_collision_norm = np.array([-np.sqrt(2) / 2., -np.sqrt(2) / 2., 0.])

    test_vnew, test_col_norm = compute_col_norm_and_new_vel_obst(pos=quad_pos, vel=quad_vel, obstacle_pos=obst_pos)
    assert np.around(test_vnew, decimals=6) == np.around(true_vnew, decimals=6)
    assert test_col_norm.all() == true_collision_norm.all()
    return


def unit_test():
    test_compute_col_norm_and_new_vel_obst()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
