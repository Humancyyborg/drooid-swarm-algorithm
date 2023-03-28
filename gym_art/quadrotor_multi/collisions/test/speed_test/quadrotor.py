import sys
import timeit
import numpy as np

from gym_art.quadrotor_multi.collisions.quadrotors import compute_col_norm_and_new_velocities
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


def normal_perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2):
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2)
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

        dyn1_new_vel_dir = np.dot(vel1 + dyn1_vel_shift, collision_norm)
        dyn2_new_vel_dir = np.dot(vel2 + dyn2_vel_shift, collision_norm)

        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            break

    # Get new vel
    max_vel_magn = max(np.linalg.norm(vel1), np.linalg.norm(vel2))
    vel1 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel1, vel_shift=dyn1_vel_shift)
    vel2 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel2, vel_shift=dyn2_vel_shift)

    # Get new omega
    new_omega = compute_new_omega()
    omega1 += new_omega
    omega2 -= new_omega

    return vel1, omega1, vel2, omega2


def test_perform_collision_between_drones():
    SETUP_CODE = '''from __main__ import normal_perform_collision_between_drones; import numpy as np'''

    TEST_CODE = '''pos1=np.array([0., 0., 0.]); vel1=np.array([0., 0., 1.]); omega1=np.array([0.5, 0.1, 0.2]); pos2=np.array([0.1, 0.05, 0.01]); vel2=np.array([0.2, 0.5, 0.1]); omega2=np.array([0.8, 0.3, 0.1]); normal_perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2)'''

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=5,
                          number=int(1e3))

    print('Speed: perform_collision_between_drones')
    print('Normal:   ', times)
    print('Mean:   ', np.mean(times))


    SETUP_CODE = '''from gym_art.quadrotor_multi.collisions.quadrotors import perform_collision_between_drones; import numpy as np'''

    TEST_CODE = '''pos1=np.array([0., 0., 0.]); vel1=np.array([0., 0., 1.]); omega1=np.array([0.5, 0.1, 0.2]); pos2=np.array([0.1, 0.05, 0.01]); vel2=np.array([0.2, 0.5, 0.1]); omega2=np.array([0.8, 0.3, 0.1]); perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2)'''

    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=6,
                          number=int(1e3))

    print('Optimized:   ', times)
    print('Mean:   ', np.mean(times[1:]))


def speed_test():
    test_perform_collision_between_drones()
    return


if __name__ == "__main__":
    sys.exit(speed_test())
