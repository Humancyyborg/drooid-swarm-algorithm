import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import EPS


@njit
def compute_new_vel(max_vel_magn, vel, vel_shift, low=0.2, high=0.8):
    vel_decay_ratio = np.random.uniform(low, high)
    vel_new = vel + vel_shift
    vel_new_mag = np.linalg.norm(vel_new)
    vel_new_dir = vel_new / (vel_new_mag + EPS if vel_new_mag == 0.0 else vel_new_mag)
    vel_new_mag = min(vel_new_mag * vel_decay_ratio, max_vel_magn)
    vel_new = vel_new_dir * vel_new_mag

    vel_shift = vel_new - vel
    vel += vel_shift
    return vel


@njit
def compute_new_omega(magn_scale=20.0):
    # Random forces for omega
    # This will amount to max 3.5 revolutions per second
    omega_max = magn_scale * np.pi
    omega = np.random.uniform(-1, 1, size=(3,))
    omega_mag = np.linalg.norm(omega)

    omega_dir = omega / (omega_mag + EPS if omega_mag == 0.0 else omega_mag)
    omega_mag = np.random.uniform(omega_max / 2, omega_max)
    omega = omega_dir * omega_mag

    return omega


if __name__ == "__main__":
    def main():
        import timeit
        SETUP_CODE = '''from __main__ import calculate_collision_matrix; import numpy as np'''

        TEST_CODE = '''calculate_collision_matrix(positions=np.ones((8, 3)), arm=0.05, hitbox_radius=2)'''

        # timeit.repeat statement
        times = timeit.repeat(setup=SETUP_CODE,
                              stmt=TEST_CODE,
                              repeat=5,
                              number=int(1e4))

        # printing minimum exec. time
        print('times:   ', times)
        print('mean times:   ', np.mean(times[1:]))


    if __name__ == '__main__':
        import sys

        sys.exit(main())
