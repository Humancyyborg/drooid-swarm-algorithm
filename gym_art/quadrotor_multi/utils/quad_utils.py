import numpy as np
import numpy.random as nr
from numba import njit
from numpy.linalg import norm
from numpy import cos, sin
from copy import deepcopy

EPS = 1e-5
QUAD_RADIUS = 0.05

SELF_OBS_REPR = {
    'xyz_vxyz_R_omega': 18,
    'xyz_vxyz_R_omega_wall': 24,
    'xyz_vxyz_R_omega_floor_ceiling': 20,
}

NEIGHBOR_OBS = {
    'none': 0,
    'pos_vel': 6,
    'pos_vel_goals': 9,
    'pos_vel_goals_ndist_gdist': 11,
}

QUAD_COLOR = (
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


# dict pretty printing
def print_dic(dic, indent=""):
    for key, item in dic.items():
        if isinstance(item, dict):
            print(indent, key + ":")
            print_dic(item, indent=indent + "  ")
        else:
            print(indent, key + ":", item)


# walk dictionary
def walk_dict(node, call):
    for key, item in node.items():
        if isinstance(item, dict):
            walk_dict(item, call)
        else:
            node[key] = call(key, item)


def walk_2dict(node1, node2, call):
    for key, item in node1.items():
        if isinstance(item, dict):
            walk_2dict(item, node2[key], call)
        else:
            node1[key], node2[key] = call(key, item, node2[key])


# numpy's cross is really slow for some reason
def cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


# returns (normalized vector, original norm)
def normalize(x):
    # n = norm(x)
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5  # np.sqrt(np.cumsum(np.square(x)))[2]

    if n < 0.00001:
        return x, 0
    return x / n, n


def norm2(x):
    return np.sum(x ** 2)


# uniformly sample from the set of all 3D rotation matrices
def rand_uniform_rot3d():
    randunit = lambda: normalize(np.random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left, _ = normalize(cross(up, fwd))
    # import pdb; pdb.set_trace()
    up = cross(fwd, left)
    rot = np.column_stack([fwd, left, up])
    return rot


# shorter way to construct a numpy array
def npa(*args):
    return np.array(args)


def clamp_norm(x, maxnorm):
    # n = np.linalg.norm(x)
    # n = np.sqrt(np.cumsum(np.square(x)))[2]
    n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5
    return x if n <= maxnorm else (maxnorm / n) * x


# project a vector into the x-y plane and normalize it.
def to_xyhat(vec):
    v = deepcopy(vec)
    v[2] = 0
    v, _ = normalize(v)
    return v


def log_error(err_str, ):
    with open("/tmp/sac/errors.txt", "a") as myfile:
        myfile.write(err_str)
        # myfile.write('###############################################')


def quat2R(qw, qx, qy, qz):
    R = \
        [[1.0 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
         [2 * qx * qy + 2 * qz * qw, 1.0 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
         [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1.0 - 2 * qx ** 2 - 2 * qy ** 2]]
    return np.array(R)


quat2R_numba = njit()(quat2R)


def qwxyz2R(quat):
    return quat2R(qw=quat[0], qx=quat[1], qy=quat[2], qz=quat[3])


def quatXquat(quat, quat_theta):
    ## quat * quat_theta
    noisy_quat = np.zeros(4)
    noisy_quat[0] = quat[0] * quat_theta[0] - quat[1] * quat_theta[1] - quat[2] * quat_theta[2] - quat[3] * quat_theta[
        3]
    noisy_quat[1] = quat[0] * quat_theta[1] + quat[1] * quat_theta[0] - quat[2] * quat_theta[3] + quat[3] * quat_theta[
        2]
    noisy_quat[2] = quat[0] * quat_theta[2] + quat[1] * quat_theta[3] + quat[2] * quat_theta[0] - quat[3] * quat_theta[
        1]
    noisy_quat[3] = quat[0] * quat_theta[3] - quat[1] * quat_theta[2] + quat[2] * quat_theta[1] + quat[3] * quat_theta[
        0]
    return noisy_quat


quatXquat_numba = njit()(quatXquat)


def R2quat(rot):
    # print('R2quat: ', rot, type(rot))
    R = rot.reshape([3, 3])
    w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    w4 = (4.0 * w)
    x = (R[2, 1] - R[1, 2]) / w4
    y = (R[0, 2] - R[2, 0]) / w4
    z = (R[1, 0] - R[0, 1]) / w4
    return np.array([w, x, y, z])


def rot2D(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def rotZ(theta):
    r = np.eye(4)
    r[:2, :2] = rot2D(theta)
    return r


def rpy2R(r, p, y):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(r), -np.sin(r)],
                    [0, np.sin(r), np.cos(r)]
                    ])
    R_y = np.array([[np.cos(p), 0, np.sin(p)],
                    [0, 1, 0],
                    [-np.sin(p), 0, np.cos(p)]
                    ])
    R_z = np.array([[np.cos(y), -np.sin(y), 0],
                    [np.sin(y), np.cos(y), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def randyaw():
    rotz = np.random.uniform(-np.pi, np.pi)
    return rotZ(rotz)[:3, :3]


def exUxe(e, U):
    """
    Cross product approximation
    exUxe = U - (U @ e) * e, where
    Args:
        e[3,1] - norm vector (assumes the same norm vector for all vectors in the batch U)
        U[3,batch_dim] - set of vectors to perform cross product on
    Returns:
        [3,batch_dim] - batch-wise cross product approximation
    """
    return U - (U.T @ rot_z).T * np.repeat(rot_z, U.shape[1], axis=1)


def cross_vec(v1, v2):
    return np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]]) @ v2


def cross_mx4(V1, V2):
    x1 = cross(V1[0, :], V2[0, :])
    x2 = cross(V1[1, :], V2[1, :])
    x3 = cross(V1[2, :], V2[2, :])
    x4 = cross(V1[3, :], V2[3, :])
    return np.array([x1, x2, x3, x4])


def cross_vec_mx4(V1, V2):
    x1 = cross(V1, V2[0, :])
    x2 = cross(V1, V2[1, :])
    x3 = cross(V1, V2[2, :])
    x4 = cross(V1, V2[3, :])
    return np.array([x1, x2, x3, x4])


def dict_update_existing(dic, dic_upd):
    for key in dic_upd.keys():
        if isinstance(dic[key], dict):
            dict_update_existing(dic[key], dic_upd[key])
        else:
            dic[key] = dic_upd[key]


@njit
def spherical_coordinate(x, y):
    return [cos(x) * cos(y), sin(x) * cos(y), sin(y)]


@njit
def generate_points(n=3):
    if n < 3:
        # print("The number of goals can not smaller than 3, The system has cast it to 3")
        n = 3

    x = 0.1 + 1.2 * n

    pts = []
    start = (-1. + 1. / (n - 1.))
    increment = (2. - 2. / (n - 1.)) / (n - 1.)
    pi = np.pi
    for j in range(n):
        s = start + j * increment
        pts.append(spherical_coordinate(
            s * x, pi / 2. * np.sign(s) * (1. - np.sqrt(1. - abs(s)))
        ))

    return pts


@njit
def get_sphere_radius(num, dist):
    A = 1.75388487222762
    B = 0.860487305801679
    C = 10.3632729642351
    D = 0.0920858134405214
    ratio = (A - D) / (1 + (num / C) ** B) + D
    radius = dist / ratio
    return radius


@njit
def get_circle_radius(num, dist):
    theta = 2 * np.pi / num
    radius = (0.5 * dist) / np.sin(theta / 2)
    return radius


@njit
def get_grid_dim_number(num):
    assert num > 0
    sqrt_goal_num = np.sqrt(num)
    grid_number = int(np.floor(sqrt_goal_num))
    dim_1 = grid_number
    while dim_1 > 1:
        if num % dim_1 == 0:
            break
        else:
            dim_1 -= 1

    dim_2 = num // dim_1
    return dim_1, dim_2


@njit
def calculate_obst_drone_proximity_penalties(distances, arm, dt, penalty_fall_off, max_penalty, num_agents):
    if not penalty_fall_off:
        # smooth penalties is disabled
        return np.zeros(num_agents)

    dist_ratio = 1 - distances / (penalty_fall_off * arm)
    dist_ratio = np.maximum(dist_ratio, 0)
    penalties = dist_ratio * max_penalty

    return dt * penalties


@njit
def compute_col_norm_and_new_velocities(dyn1_pos, dyn2_pos, dyn1_vel, dyn2_vel):
    # Ge the collision normal, i.e difference in position
    collision_norm = dyn1_pos - dyn2_pos
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    v1new = np.dot(dyn1_vel, collision_norm)
    v2new = np.dot(dyn2_vel, collision_norm)

    return v1new, v2new, collision_norm


@njit
def compute_col_norm_and_new_vel_obst(quad_pos, quad_vel, obstacle_pos):
    collision_norm = quad_pos - obstacle_pos
    # difference in z position is 0, given obstacle height is same as room height
    collision_norm[2] = 0.0
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    vnew = np.dot(quad_vel, collision_norm)

    return vnew, collision_norm


@njit
def compute_new_vel(max_vel_magn, vel, vel_shift, coeff, vel_decay_ratio):
    vel_new = vel + vel_shift
    vel_new_mag = np.linalg.norm(vel_new)
    vel_new_dir = vel_new / (vel_new_mag + EPS if vel_new_mag == 0.0 else vel_new_mag)
    vel_new_mag = min(vel_new_mag * vel_decay_ratio, max_vel_magn)
    vel_new = vel_new_dir * vel_new_mag

    vel_shift = vel_new - vel
    vel += vel_shift * coeff
    return vel


@njit
def compute_new_omega():
    # Random forces for omega
    # This will amount to max 3.5 revolutions per second
    omega_max = 20 * np.pi
    omega = np.random.uniform(-1.0, 1.0, size=3)
    omega_mag = np.linalg.norm(omega)

    omega_dir = omega / (omega_mag + EPS if omega_mag == 0.0 else omega_mag)
    omega_mag = np.random.uniform(omega_max / 2, omega_max)
    omega = omega_dir * omega_mag

    return omega


# This function is to change the velocities after a collision happens between two bodies
def perform_collision_between_drones(dyn1, dyn2, col_coeff=1.0):
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(
        dyn1_pos=dyn1.pos, dyn2_pos=dyn2.pos, dyn1_vel=dyn1.vel, dyn2_vel=dyn2.vel)
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

        dyn1_new_vel_dir = np.dot(dyn1.vel + dyn1_vel_shift, collision_norm)
        dyn2_new_vel_dir = np.dot(dyn2.vel + dyn2_vel_shift, collision_norm)

        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            break

    # Get new vel
    max_vel_magn = max(np.linalg.norm(dyn1.vel), np.linalg.norm(dyn2.vel))

    vel_decay_ratio = np.random.uniform(low=0.2, high=0.8, size=2)
    dyn1.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=dyn1.vel, vel_shift=dyn1_vel_shift, coeff=col_coeff,
                               vel_decay_ratio=vel_decay_ratio[0])
    dyn2.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=dyn2.vel, vel_shift=dyn2_vel_shift, coeff=col_coeff,
                               vel_decay_ratio=vel_decay_ratio[1])

    # Get new omega
    new_omega = compute_new_omega()
    dyn1.omega += new_omega * col_coeff
    dyn2.omega -= new_omega * col_coeff


def perform_collision_with_obstacle(drone_dyn, obstacle_pos, obstacle_size, col_coeff=1.0):
    # Vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    vnew, collision_norm = compute_col_norm_and_new_vel_obst(quad_pos=drone_dyn.pos, quad_vel=drone_dyn.vel,
                                                             obstacle_pos=obstacle_pos)
    vel_change = -vnew * collision_norm

    dyn_vel_shift = vel_change
    for _ in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        vel_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        dyn_vel_shift = vel_change + vel_noise
        if np.dot(drone_dyn.vel + dyn_vel_shift, collision_norm) > 0:
            break

    max_vel_magn = np.linalg.norm(drone_dyn.vel)
    if np.linalg.norm(drone_dyn.pos - obstacle_pos) <= obstacle_size:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel, vel_shift=dyn_vel_shift,
                                        coeff=col_coeff, vel_decay_ratio=1.0)
    else:
        vel_decay_ratio = np.random.uniform(low=0.2, high=0.8)
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel, vel_shift=dyn_vel_shift,
                                        coeff=col_coeff, vel_decay_ratio=vel_decay_ratio)

    # Random forces for omega
    new_omega = compute_new_omega()
    drone_dyn.omega += new_omega * col_coeff


def perform_collision_with_wall(drone_dyn, room_box, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    # Decrease drone's speed after collision with wall
    drone_speed = np.linalg.norm(drone_dyn.vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    drone_pos = drone_dyn.pos
    x_list = [drone_pos[0] == room_box[0][0], drone_pos[0] == room_box[1][0]]
    y_list = [drone_pos[1] == room_box[0][1], drone_pos[1] == room_box[1][1]]

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    if x_list[0]:
        direction[0] = np.random.uniform(low=0.1, high=1.0)
    elif x_list[1]:
        direction[0] = np.random.uniform(low=-1.0, high=-0.1)

    if y_list[0]:
        direction[1] = np.random.uniform(low=0.1, high=1.0)
    elif y_list[1]:
        direction[1] = np.random.uniform(low=-1.0, high=-0.1)

    direction[2] = np.random.uniform(low=-1.0, high=-0.5)

    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    new_omega = compute_new_omega()
    drone_dyn.omega += new_omega


def perform_collision_with_ceiling(drone_dyn, damp_low_speed_ratio=0.2, damp_high_speed_ratio=0.8,
                                   lowest_speed=0.1, highest_speed=6.0, eps=1e-5):
    drone_speed = np.linalg.norm(drone_dyn.vel)
    real_speed = np.random.uniform(low=damp_low_speed_ratio * drone_speed, high=damp_high_speed_ratio * drone_speed)
    real_speed = np.clip(real_speed, a_min=lowest_speed, a_max=highest_speed)

    direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
    direction[2] = np.random.uniform(low=-1.0, high=-0.5)
    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / (direction_mag + eps)

    drone_dyn.vel = real_speed * direction_norm

    # Random forces for omega
    new_omega = compute_new_omega()
    drone_dyn.omega += new_omega


def get_vel_omega_norm(z_axis):
    # vel_norm
    noise_z_axis = z_axis + np.random.uniform(low=-0.1, high=0.1, size=3)
    noise_z_axis_mag = np.linalg.norm(noise_z_axis)
    noise_z_axis_norm = noise_z_axis / (noise_z_axis_mag + EPS if noise_z_axis_mag == 0.0 else noise_z_axis_mag)
    down_z_axis_norm = -1.0 * noise_z_axis_norm

    # omega norm
    dir_omega = np.random.uniform(low=-1, high=1, size=3)
    dir_omega_mag = np.linalg.norm(dir_omega)
    dir_omega_norm = dir_omega / (dir_omega_mag + EPS if dir_omega_mag == 0.0 else dir_omega_mag)

    return down_z_axis_norm, dir_omega_norm


def perform_downwash(drones_dyn, dt):
    # based on some data from Neural-Swarm: https://arxiv.org/pdf/2003.02992.pdf, Fig. 3
    # quadrotor weights: 34 grams
    # 0.5 m, force = 4 grams ; 0.4 m, force = 6 grams
    # 0.3 m, force = 8 grams ; 0.2 m, force = 10 grams
    # force function: f(x) = -20x + 14
    # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17, x in [0, 0.7]
    # Use cylinder to simulate the downwash area
    # The downwash area is a cylinder with radius of 2 arm ~ 10 cm and height of 1.0 m
    xy_downwash = 0.1
    z_downwash = 0.7
    # get pos
    dyns_pos = np.array([d.pos for d in drones_dyn])
    # get z_axis
    dyns_z_axis = np.array([d.rot[:, -1] for d in drones_dyn])

    # drone num
    dyns_num = len(drones_dyn)
    # check if neighbors drones are within the downwash areas, if yes, apply downwash
    for i in range(dyns_num):
        z_axis = dyns_z_axis[i]
        neighbor_pos = dyns_pos - dyns_pos[i]
        neighbor_pos_dist = np.linalg.norm(neighbor_pos, axis=1)
        # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17
        # x in [0, 0.7], a(x) in [0.0, 0.41]
        # acc = (1 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.03, high=0.03)
        acc = (6 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.1, high=0.1)
        acc = np.maximum(EPS, acc)

        # omega downwash given neighbor_pos_dist
        # 0.3 * (x - 1)^2 + random(-0.01, 0.01)
        omega_downwash = 0.3 * (neighbor_pos_dist - 1) ** 2 + np.random.uniform(low=-0.01, high=0.01)
        omega_downwash = np.maximum(EPS, omega_downwash)

        rel_dists_z = np.dot(neighbor_pos, z_axis)
        rel_dists_xy = np.sqrt(neighbor_pos_dist ** 2 - rel_dists_z ** 2)

        for j in range(dyns_num):
            if i == j:
                continue

            if -z_downwash < rel_dists_z[j] < 0 and rel_dists_xy[j] < xy_downwash:
                down_z_axis_norm, dir_omega_norm = get_vel_omega_norm(z_axis=z_axis)
                drones_dyn[j].vel += acc[j] * down_z_axis_norm * dt
                drones_dyn[j].omega += omega_downwash[j] * dir_omega_norm * dt

    return


class OUNoise:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3, use_seed=False):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        @param: use_seed: set the random number generator to some specific seed for test
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        if use_seed:
            nr.seed(2)

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state


if __name__ == "__main__":
    """
        measure time (s)

        1) generate_points; n=8
            numba: mean: 0.254, std: 0.00295
            plain: mean: 8.486., std: 0.0499
        
        2) get_sphere_radius: num=8; dist=0.5
            numba: mean: 0.037, std: 0.000586
            plain: mean: 0.045., std: 0.00075
        
        3) get_circle_radius
            stmt = 'get_circle_radius(num, dist)'
            setup = 'from __main__ import get_circle_radius; import numpy as np; ' \
            'num=8; dist=0.65'
            
            numba: mean: 0.031, std: 0.000186
            plain: mean: 0.191, std: 0.005676
        
        
        4) get_grid_dim_number: num=8
        numba: mean: 0.0292, std: 0.00041
        plain: mean: 0.3788, std: 0.00556
        
        5) calculate_obst_drone_proximity_penalties 
        stmt = 'calculate_obst_drone_proximity_penalties(distances, arm, dt, 
        penalty_fall_off, max_penalty, num_agents)' setup = 'from __main__ import 
        calculate_obst_drone_proximity_penalties; import numpy as np; distances=np.random.uniform(low=0.01, 
        high=10.0, size=(8,8)); arm=0.05; dt=0.01; penalty_fall_off=0.2; max_penalty=10.0; num_agents=8'
        
        numba: mean: 0.171, std: 0.0009
        plain: mean: 1.0845, std: 0.013
        
        6) compute_col_norm_and_new_velocities(dyn1_pos, dyn2_pos, dyn1_vel, dyn2_vel)
            stmt = 'compute_col_norm_and_new_velocities(dyn1_pos, dyn2_pos, dyn1_vel, dyn2_vel)'
            setup = 'from __main__ import compute_col_norm_and_new_velocities; import numpy as np; ' \
            'dyn1_pos=np.zeros(3); dyn2_pos=np.ones(3); dyn1_vel=np.ones(3) * 0.11; dyn2_vel=np.array([1.0, 0.8, 0.655])'
        
            numba: mean: 0.1952, std: 0.00077
            plain: mean: 1.2963, std: 0.00633
        
        7) compute_col_norm_and_new_vel_obst(quad_pos, quad_vel, obstacle_pos)
            stmt = 'compute_col_norm_and_new_vel_obst(quad_pos, quad_vel, obstacle_pos)'
            setup = 'from __main__ import compute_col_norm_and_new_vel_obst; import numpy as np; ' \
            'quad_pos=np.zeros(3); quad_vel=np.ones(3); obstacle_pos=np.ones(3) * 0.11'
            
            numba: mean: 0.1848, std: 0.0021
            plain: mean: 1.1344, std: 0.0117
        
        8) compute_new_vel
        stmt = 'compute_new_vel(max_vel_magn, vel, vel_shift, coeff, low=0.2, high=0.8)'
        setup = 'from __main__ import compute_new_vel; import numpy as np; ' \
            'max_vel_magn=5; vel=np.array([1., 2., 0.5]); vel_shift=np.array([0.2, 0.5, 0.1]); coeff=1.0'
        
        numba: mean: 0.1844, std: 0.0032
        plain: mean: 1.7305, std: 0.0068
        
        9) compute_new_omega: 1e5: ~ 14 faster
            stmt = 'compute_new_omega()'
            setup = 'from __main__ import compute_new_omega; import numpy as np'
            
            numba: mean: 0.06838, std: 0.0014
            plain: mean: 0.95835, std: 0.0107
        
    """
    """
    import timeit

    stmt = 'compute_col_norm_and_new_vel_obst(quad_pos, quad_vel, obstacle_pos)'
    setup = 'from __main__ import compute_col_norm_and_new_vel_obst; import numpy as np; ' \
            'quad_pos=np.zeros(3); quad_vel=np.ones(3); obstacle_pos=np.ones(3) * 0.11'
    use_numba = True
    # Pass the argument 'n=100' to my_function() and time it
    if use_numba:
        repeat = 6
    else:
        repeat = 5

    t = timeit.Timer(stmt=stmt, setup=setup)
    time_taken = t.repeat(repeat=repeat, number=int(2e5))
    if use_numba:
        time_taken = np.array(time_taken[1:])
    else:
        time_taken = np.array(time_taken)

    print('Time taken:', time_taken)
    print('Time Mean:', time_taken.mean())
    print('Time Std:', time_taken.std())
    """

    # Cross product test
    import time

    rot_z = np.array([[3], [4], [5]])
    rot_z = rot_z / np.linalg.norm(rot_z)
    v_rotors = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 8, 7, 6]])

    start_time = time.time()
    cr1 = v_rotors - (v_rotors.T @ rot_z).T * np.repeat(rot_z, 4, axis=1)
    print("cr1 time:", time.time() - start_time)

    start_time = time.time()
    cr2 = np.cross(rot_z.T, np.cross(v_rotors.T, np.repeat(rot_z, 4, axis=1).T)).T
    print("cr2 time:", time.time() - start_time)
    print("cr1 == cr2:", np.sum(cr1 - cr2) < 1e-10)