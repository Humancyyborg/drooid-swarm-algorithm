import numpy as np

quad_color = (
    (1.0, 0.0, 0.0),  # red
    (0.4, 0.4, 0.4),  # darkgrey
    (0.0, 1.0, 0.0),  # green
    (0.0, 0.0, 1.0),  # blue
    (1.0, 1.0, 0.0),  # yellow
    (0.0, 1.0, 1.0),  # cyan
    (1.0, 0.0, 1.0),  # magenta
    (0.5, 0.0, 0.0),  # darkred
    (0.0, 0.5, 0.0),  # darkgreen
    (0.0, 0.0, 0.5),  # darkblue
    (0.0, 0.5, 0.5),  # darkcyan
    (0.5, 0.0, 0.5),  # darkmagenta
    (0.5, 0.5, 0.0),  # darkyellow
    (0.8, 0.8, 0.8),  # lightgrey
    (1.0, 0.0, 1.0),  # Violet
)

obs_self_size_dict = {
    'xyz_vxyz_R_omega': 18,
    'xyz_vxyz_R_omega_floor': 19,
    'xyz_vxyz_R_omega_floor_cwallid_cwall': 21,
    'xyz_vxyz_R_omega_wall': 24
}

obs_neighbor_size_dict = {
    'none': 0,
    'pos_vel': 6,
    'pos_vel_goals': 9,
    'pos_vel_goals_ndist_gdist': 11,
}

obs_obst_size_dict = {
    'none': 0,
    'cpoint': 2,
    'posxy_size': 3,
    'pos_size': 4,
    'pos_vel_size': 7,
    'pos_vel_size_shape': 8
}

quad_arm_size = 0.04596194077712559

test_vel = np.array([0.0, 1.0, 0.0])
test_omega = np.array([0.0, 0.0, 0.0])
test_start_pos = np.array([0.0, -3.3, 2.0])
test_end_pos = np.array([0.0, -2.8, 2.0])
test_other_drone_pos = np.array([-4.5, 0.0, 3.0])
test_rot = np.identity(3)
test_obst_pp = np.array([-3.9, -3.0, 5.0])