#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236

Developers:
James Preiss, Artem Molchanov, Tao Chen 

References:
[1] RotorS: https://www.researchgate.net/profile/Fadri_Furrer/publication/309291237_RotorS_-_A_Modular_Gazebo_MAV_Simulator_Framework/links/5a0169c4a6fdcc82a3183f8f/RotorS-A-Modular-Gazebo-MAV-Simulator-Framework.pdf
[2] CrazyFlie modelling: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
[3] HummingBird: http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/
[4] CrazyFlie thrusters transition functions: https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
[5] HummingBird modelling: https://digitalrepository.unm.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1189&context=ece_etds
[6] Rotation b/w matrices: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
[7] Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
"""
import copy
import logging

import gym_art.quadrotor_multi.get_state as get_state
import gym_art.quadrotor_multi.quadrotor_randomization as quad_rand
# MATH
# GYM
from gym.utils import seeding
from gym_art.quadrotor_multi.inertia import QuadLink, QuadLinkSimplified
from gym_art.quadrotor_multi.numba_utils import *
from gym_art.quadrotor_multi.quadrotor_control import *
from gym_art.quadrotor_multi.sensor_noise import SensorNoise
# Numba
from numba import njit

logger = logging.getLogger(__name__)

GRAV = 9.81  # default gravitational constant
EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


# WARN:
# - linearity is set to 1 always, by means of check_quad_param_limits().
# The def. value of linarity for CF is set to 1 as well (due to firmware nonlinearity compensation)

def random_state(box, vel_max=15.0, omega_max=2 * np.pi):
    box = np.array(box)
    pos = np.random.uniform(low=-box, high=box, size=(3,))

    vel = np.random.uniform(low=-vel_max, high=vel_max, size=(3,))
    vel_magn = np.random.uniform(low=0., high=vel_max)
    vel = vel_magn / (np.linalg.norm(vel) + EPS) * vel

    omega = np.random.uniform(low=-omega_max, high=omega_max, size=(3,))
    omega_magn = np.random.uniform(low=0., high=omega_max)
    omega = omega_magn / (np.linalg.norm(omega) + EPS) * omega

    rot = rand_uniform_rot3d()
    return pos, vel, rot, omega


class QuadrotorDynamics:
    """
    Simple simulation of quadrotor dynamics.
    mass unit: kilogram
    arm_length unit: meter
    inertia unit: kg * m^2, 3-element vector representing diagonal matrix
    thrust_to_weight is the total, it will be divided among the 4 props
    torque_to_thrust is ratio of torque produced by prop to thrust
    thrust_noise_ratio is noise2signal ratio of the thrust noise, Ex: 0.05 = 5% of the current signal
      It is an approximate ratio, i.e. the upper bound could still be higher, due to how OU noise operates
    Coord frames: x configuration:
     - x axis between arms looking forward [x - configuration]
     - y axis pointing to the left
     - z axis up
    TODO:
    - only diagonal inertia is used at the moment
    """

    def __init__(self, model_params, room_box=None, dynamics_steps_num=1, dt=0.005, dim_mode="3D", gravity=GRAV,
                 dynamics_simplification=False, use_numba=False):

        # Dynamics
        self.pos = None
        self.vel = None
        self.rot = None
        self.omega = None
        self.thrusts = None
        self.acc = None
        self.accelerometer = None

        self.dynamics_steps_num = dynamics_steps_num
        self.dynamics_simplification = dynamics_simplification
        # cw = 1 ; ccw = -1 [ccw, cw, ccw, cw]
        # Reference: https://docs.google.com/document/d/1wZMZQ6jilDbj0JtfeYt0TonjxoMPIgHwYbrFrMNls84/edit
        self.prop_ccw = np.array([-1., 1., -1., 1.])
        self.omega_max = 40.  # rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        self.vxyz_max = 3.  # m/s
        self.gravity = gravity
        self.acc_max = 3. * GRAV

        # Internal State variables
        self.since_last_ort_check = 0  # counter
        self.since_last_ort_check_limit = 0.04  # when to check for non-orthogonality

        self.rot_nonort_limit = 0.01  # How much of non-orthogonality in the R matrix to tolerate
        self.rot_nonort_coeff_maxsofar = 0.  # Statistics on the max number of nonorthogonality that we had

        self.since_last_svd = 0  # counter
        self.since_last_svd_limit = 0.5  # in sec - how often mandatory orthogonalization should be applied

        self.eye = np.eye(3)

        # Initializing model
        self.thrust_noise = None
        self.model = None
        self.mass = None
        self.thrust_to_weight = None
        self.torque_to_thrust = None
        self.motor_linearity = None
        self.C_rot_drag = None
        self.C_rot_roll = None
        self.motor_damp_time_up = 0.15
        self.motor_damp_time_down = 0.15
        self.thrust_noise_ratio = None
        self.vel_damp = None
        self.damp_omega_quadratic = None
        self.motor_assymetry = None
        self.thrust_max = None
        self.torque_max = None
        self.prop_pos = None
        self.prop_crossproducts = None
        self.prop_ccw_mx = None
        self.G_omega_thrust = None
        self.C_omega_prop = None
        self.G_omega = None
        self.thrust_sum_mx = None
        self.arm = None
        self.torque_to_inertia = None

        self.update_model(model_params)

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * dt / (self.motor_damp_time_down + EPS)
        self.motor_tau = self.motor_tau_up * np.ones([4, ])

        # Momentum
        self.thrust_cmds_damp = np.zeros([4])
        self.thrust_rot_damp = np.zeros([4])

        # Sanity checks
        assert self.inertia.shape == (3,)

        # OTHER PARAMETERS
        if room_box is None:
            self.room_box = np.array([[0., 0., 0.], [10., 10., 10.]])
        else:
            self.room_box = np.array(room_box).copy()

        # Selecting 1D, Planar or Full 3D modes
        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.control_mx = np.ones([4, 1])
        elif self.dim_mode == '2D':
            self.control_mx = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        elif self.dim_mode == '3D':
            self.control_mx = np.eye(4)
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)

        # # Identify if drone is on the floor
        self.on_floor = False
        # # If pos_z smaller than this threshold, we assume that drone collide with the floor
        self.floor_threshold = 0.05
        # # Friction coefficient
        self.mu = 0.6

        # #Collision with room
        self.crashed_wall = False
        self.crashed_ceiling = False
        self.crashed_floor = False

        # # Numba
        self.use_numba = use_numba

    @staticmethod
    def angvel2thrust(w, linearity=0.424):
        """
        CrazyFlie: linearity=0.424
        Args:
            w: thrust_cmds_damp
            linearity (float): linearity factor factor [0 .. 1].
        """
        return (1 - linearity) * w ** 2 + linearity * w

    def update_model(self, model_params):
        if self.dynamics_simplification:
            self.model = QuadLinkSimplified(params=model_params["geom"])
        else:
            self.model = QuadLink(params=model_params["geom"])

        # PARAMETERS FOR RANDOMIZATION
        self.mass = self.model.m
        self.inertia = np.diagonal(self.model.I_com)

        self.thrust_to_weight = model_params["motor"]["thrust_to_weight"]
        self.torque_to_thrust = model_params["motor"]["torque_to_thrust"]
        self.motor_linearity = model_params["motor"]["linearity"]
        self.C_rot_drag = model_params["motor"]["C_drag"]
        self.C_rot_roll = model_params["motor"]["C_roll"]
        self.motor_damp_time_up = model_params["motor"]["damp_time_up"]
        self.motor_damp_time_down = model_params["motor"]["damp_time_down"]

        self.thrust_noise_ratio = model_params["noise"]["thrust_noise_ratio"]
        self.vel_damp = model_params["damp"]["vel"]
        self.damp_omega_quadratic = model_params["damp"]["omega_quadratic"]

        # COMPUTED (Dependent) PARAMETERS
        try:
            self.motor_assymetry = np.array(model_params["motor"]["assymetry"])
        except:
            self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
            print("WARNING: Motor assymetry was not setup. Setting assymetry to:", self.motor_assymetry)
        self.motor_assymetry = self.motor_assymetry * 4. / np.sum(self.motor_assymetry)  # re-normalizing to sum-up to 4
        self.thrust_max = GRAV * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.torque_max = self.torque_to_thrust * self.thrust_max  # propeller torque scales

        # Propeller positions in X configurations
        self.prop_pos = self.model.prop_pos

        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        self.prop_ccw_mx = np.zeros([3, 4])  # Matrix allows using matrix multiplication
        self.prop_ccw_mx[2, :] = self.prop_ccw

        # Forced dynamics auxiliary matrices
        # Prop crossproduct give torque directions
        self.G_omega_thrust = self.thrust_max * self.prop_crossproducts.T  # [3,4] @ [4,1]
        # additional torques along z-axis caused by propeller rotations
        self.C_omega_prop = self.torque_max * self.prop_ccw_mx  # [3,4] @ [4,1] = [3,1]
        self.G_omega = (1.0 / self.inertia)[:, None] * (self.G_omega_thrust + self.C_omega_prop)

        # Allows to sum-up thrusts as a linear matrix operation
        self.thrust_sum_mx = np.zeros([3, 4])  # [0,0,F_sum].T
        self.thrust_sum_mx[2, :] = 1  # [0,0,F_sum].T

        self.init_thrust_noise()

        self.arm = np.linalg.norm(self.model.motor_xyz[:2])

        # the ratio between max torque and inertia around each axis
        # the 0-1 matrix on the right is the way to sum-up
        self.torque_to_inertia = self.G_omega @ np.array([[0., 0., 0.], [0., 1., 1.], [1., 1., 0.], [1., 0., 1.]])
        self.torque_to_inertia = np.sum(self.torque_to_inertia, axis=1)
        self.reset()

    def init_thrust_noise(self):
        # sigma = 0.2 gives roughly max noise of -1 .. 1
        if self.use_numba:
            self.thrust_noise = OUNoiseNumba(4, sigma=0.2 * self.thrust_noise_ratio)
        else:
            self.thrust_noise = OUNoise(4, sigma=0.2 * self.thrust_noise_ratio)

    # pos, vel, in world coords (meters)
    # rotation is 3x3 matrix (body coords) -> (world coords)dt
    # omega is angular velocity (radians/sec) in body coords, i.e. the gyroscope
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3, 3)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.acc = np.zeros(3)
        self.accelerometer = np.array([0, 0, GRAV])
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega.astype(np.float32))
        self.thrusts = deepcopy(thrusts)

    # generate a random state (meters, meters/sec, radians/sec)

    def step(self, thrust_cmds, dt):
        thrust_noise = self.thrust_noise.noise()

        if self.use_numba:
            [self.step1_numba(thrust_cmds, dt, thrust_noise) for _ in range(self.dynamics_steps_num)]
        else:
            [self.step1(thrust_cmds, dt, thrust_noise) for _ in range(self.dynamics_steps_num)]

    # Step function integrates based on current derivative values (best fits affine dynamics model)
    # thrust_cmds is motor thrusts given in normalized range [0, 1].
    # 1 represents the max possible thrust of the motor.
    # Frames:
    # pos - global
    # vel - global
    # rot - global
    # omega - body frame
    # goal_pos - global
    def step1(self, thrust_cmds, dt, thrust_noise):
        thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)
        # Filtering the thruster and adding noise
        motor_tau = copy.deepcopy(self.motor_tau)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = copy.deepcopy(self.motor_tau_down)
        motor_tau[motor_tau > 1.] = 1.

        # Since NN commands thrusts we need to convert to rot vel and back
        # WARNING: Unfortunately if the linearity != 1 then filtering using square root is not quite correct
        # since it likely means that you are using rotational velocities as an input instead of the thrust and hence
        # you are filtering square roots of angular velocities
        thrust_rot = thrust_cmds ** 0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        # Adding noise
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)

        thrusts = self.thrust_max * self.angvel2thrust(self.thrust_cmds_damp, linearity=self.motor_linearity)
        # Prop crossproduct give torque directions
        torques = self.prop_crossproducts * thrusts[:, None]  # (4,3)=(props, xyz)

        # additional torques along z-axis caused by propeller rotations
        torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp

        # net torque: sum over propellers
        thrust_torque = np.sum(torques, axis=0)

        # Rotor drag and Rolling forces and moments
        # See Ref[1] Sec:2.1 for detailes

        # self.C_rot_drag = 0.0028
        # self.C_rot_roll = 0.003 # 0.0003
        if self.C_rot_drag != 0 or self.C_rot_roll != 0:
            # self.vel = np.zeros_like(self.vel)
            # v_rotors[3,4]  = (rot[3,3] @ vel[3,])[3,] + (omega[3,] x prop_pos[4,3])[4,3]
            # v_rotors = self.rot.T @ self.vel + np.cross(self.omega, self.model.prop_pos)
            vel_body = self.rot.T @ self.vel
            v_rotors = vel_body + cross_vec_mx4(self.omega, self.model.prop_pos)
            # assert v_rotors.shape == (4,3)
            v_rotors[:, 2] = 0.  # Projection to the rotor plane

            # Drag/Roll of rotors (both in body frame)
            rotor_drag_fi = - self.C_rot_drag * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotors  # [4,3]
            rotor_drag_force = np.sum(rotor_drag_fi, axis=0)
            # rotor_drag_ti = np.cross(rotor_drag_fi, self.model.prop_pos)#[4,3] x [4,3]
            rotor_drag_ti = cross_mx4(rotor_drag_fi, self.model.prop_pos)  # [4,3] x [4,3]
            rotor_drag_torque = np.sum(rotor_drag_ti, axis=0)

            rotor_roll_torque = \
                - self.C_rot_roll * self.prop_ccw[:, None] * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotors  # [4,3]
            rotor_roll_torque = np.sum(rotor_roll_torque, axis=0)
            rotor_visc_torque = rotor_drag_torque + rotor_roll_torque

            # Constraints (prevent numerical instabilities)
            vel_norm = np.linalg.norm(vel_body)
            rdf_norm = np.linalg.norm(rotor_drag_force)
            rdf_norm_clip = np.clip(rdf_norm, a_min=0., a_max=vel_norm * self.mass / (2 * dt))
            if rdf_norm > EPS:
                rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip

            # omega_norm = np.linalg.norm(self.omega)
            rvt_norm = np.linalg.norm(rotor_visc_torque)
            rvt_norm_clipped = np.clip(rvt_norm, a_min=0., a_max=np.linalg.norm(self.omega * self.inertia) / (2 * dt))
            if rvt_norm > EPS:
                rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped
        else:
            rotor_visc_torque = rotor_drag_force = np.zeros(3)

        # (Square) Damping using torques (in case we would like to add damping using torques)
        # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
        torque = thrust_torque + rotor_visc_torque
        thrust = npa(0, 0, np.sum(thrusts))

        # ROTATIONAL DYNAMICS
        # # Integrating rotations (based on current values)
        omega_vec = np.matmul(self.rot, self.omega)  # Change from body2world frame
        wx, wy, wz = omega_vec
        omega_norm = np.linalg.norm(omega_vec)
        if omega_norm != 0:
            # See [7]
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            dRdt = self.eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
            self.rot = dRdt @ self.rot

        # # SVD is not strictly required anymore. Performing it rarely, just in case
        self.since_last_svd += dt
        if self.since_last_svd > self.since_last_svd_limit:
            # # Perform SVD orthogonolization
            u, s, v = np.linalg.svd(self.rot)
            self.rot = np.matmul(u, v)
            self.since_last_svd = 0

        # COMPUTING OMEGA UPDATE
        # Damping using velocities (I find it more stable numerically)
        # Linear damping
        # This is only for linear damping of angular velocity.
        omega_dot = ((1.0 / self.inertia) * (cross(-self.omega, self.inertia * self.omega) + torque))

        # Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * self.omega ** 2, a_min=0.0, a_max=1.0)
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
        self.omega = np.clip(self.omega, a_min=-self.omega_max, a_max=self.omega_max)

        # TRANSLATIONAL DYNAMICS
        # Computing position
        self.pos = self.pos + dt * self.vel

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        pos_before_clip = self.pos.copy()
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        self.crashed_wall = not np.array_equal(pos_before_clip[:2], self.pos[:2])
        self.crashed_ceiling = pos_before_clip[2] > self.pos[2]

        sum_thr_drag = thrust + rotor_drag_force

        # We would get acc
        self.floor_interaction(sum_thr_drag=sum_thr_drag)

        # Computing velocities
        self.vel = (1.0 - self.vel_damp) * self.vel + dt * self.acc

        # Accelerometer measures so called "proper acceleration"
        # that includes gravity with the opposite sign
        self.accelerometer = np.matmul(self.rot.T, self.acc + [0, 0, self.gravity])

    def step1_numba(self, thrust_cmds, dt, thrust_noise):
        self.thrust_rot_damp, self.thrust_cmds_damp, self.rot, self.since_last_svd, self.omega, self.pos, thrust, \
            rotor_drag_force, self.vel = \
            calculate_torque_integrate_rotations_and_update_omega(
                thrust_cmds=thrust_cmds, dt=dt, motor_tau_up=self.motor_tau_up,
                motor_tau_down=self.motor_tau_down, thrust_cmds_damp=self.thrust_cmds_damp,
                thrust_rot_damp=self.thrust_rot_damp, thr_noise=thrust_noise, thrust_max=self.thrust_max,
                motor_linearity=self.motor_linearity, prop_crossproducts=self.prop_crossproducts,
                prop_ccw=self.prop_ccw, torque_max=self.torque_max, rot=self.rot, omega=np.float64(self.omega),
                eye=self.eye, since_last_svd=self.since_last_svd, since_last_svd_limit=self.since_last_svd_limit,
                inertia=self.inertia, damp_omega_quadratic=self.damp_omega_quadratic, omega_max=self.omega_max,
                pos=self.pos, vel=self.vel)

        pos_before_clip = self.pos.copy()
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        # Detect collision with walls
        self.crashed_wall = not np.array_equal(pos_before_clip[:2], self.pos[:2])
        self.crashed_ceiling = pos_before_clip[2] > self.pos[2]

        # Set constant variables up for numba
        sum_thr_drag = thrust + rotor_drag_force
        grav_arr = np.float64([0, 0, self.gravity])

        self.pos, self.vel, self.acc, self.omega, self.rot, self.thrust_cmds_damp, self.thrust_rot_damp, \
            self.on_floor, self.crashed_floor = floor_interaction_numba(
                pos=self.pos, vel=self.vel, rot=self.rot, omega=self.omega, mu=self.mu, mass=self.mass,
                sum_thr_drag=sum_thr_drag, thrust_cmds_damp=self.thrust_cmds_damp, thrust_rot_damp=self.thrust_rot_damp,
                floor_threshold=self.floor_threshold, on_floor=self.on_floor)

        self.vel, self.accelerometer = compute_velocity_and_acceleration(
            vel=self.vel, vel_damp=self.vel_damp, dt=dt, rot_tpose=self.rot.T, grav_arr=grav_arr, acc=self.acc)

    def reset(self):
        self.thrust_cmds_damp = np.zeros([4])
        self.thrust_rot_damp = np.zeros([4])

    def floor_interaction(self, sum_thr_drag):
        # Change pos, omega, rot, acc
        self.crashed_floor = False
        if self.pos[2] <= self.floor_threshold:
            self.pos = np.array((self.pos[0], self.pos[1], self.floor_threshold))
            force = np.matmul(self.rot, sum_thr_drag)
            if self.on_floor:
                # Drone is on the floor, and on_floor flag still True
                theta = np.arctan2(self.rot[1][0], self.rot[0][0] + EPS)
                c, s = np.cos(theta), np.sin(theta)
                self.rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

                # Add friction if drone is on the floor
                f = self.mu * GRAV * npa(np.sign(force[0]), np.sign(force[1]), 0) * self.mass
                # Since fiction cannot be greater than force, we need to clip it
                for i in range(2):
                    if np.abs(f[i]) > np.abs(force[i]):
                        f[i] = force[i]
                force -= f

            else:
                # Previous step, drone still in the air, but in this step, it hits the floor
                # In previous step, self.on_floor = False, self.crashed_floor = False
                self.on_floor = True
                self.crashed_floor = True
                # Set vel to [0, 0, 0]
                self.vel, self.acc = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
                self.omega = np.zeros(3, dtype=np.float32)
                # Set rot
                theta = np.arctan2(self.rot[1][0], self.rot[0][0] + EPS)
                c, s = np.cos(theta), np.sin(theta)
                if self.rot[2, 2] < 0:
                    self.rot = randyaw()
                    while np.dot(self.rot[:, 0], to_xyhat(-self.pos)) < 0.5:
                        self.rot = randyaw()
                else:
                    self.rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

                self.set_state(self.pos, self.vel, self.rot, self.omega)

                # reset momentum / accumulation of thrust
                self.thrust_cmds_damp = np.zeros([4])
                self.thrust_rot_damp = np.zeros([4])

            self.acc = [0., 0., -GRAV] + (1.0 / self.mass) * force
            self.acc[2] = np.maximum(0, self.acc[2])
        else:
            # self.pos[2] > self.floor_threshold
            if self.on_floor:
                # Drone is in the air, while on_floor flag still True
                self.on_floor = False

            # Computing accelerations
            force = np.matmul(self.rot, sum_thr_drag)
            self.acc = [0., 0., -GRAV] + (1.0 / self.mass) * force

    def state_vector(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

    @staticmethod
    def action_space():
        low = np.zeros(4)
        high = np.ones(4)
        return spaces.Box(low, high, dtype=np.float32)

    def __deepcopy__(self, memo):
        """Certain numba-optimized instance attributes can't be naively copied."""

        cls = self.__class__
        copied_dynamics = cls.__new__(cls)
        memo[id(self)] = copied_dynamics

        skip_copying = {"thrust_noise"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_dynamics, k, deepcopy(v, memo))

        copied_dynamics.init_thrust_noise()
        return copied_dynamics


def compute_reward_weighted(dynamics, goal, action, dt, time_remain, rew_coeff, action_prev, on_floor=False,
                            log_all_info=False):
    dist = np.linalg.norm(goal - dynamics.pos)
    cost_pos_raw = dist
    cost_pos = rew_coeff["pos"] * cost_pos_raw

    # penalize amount of control effort
    cost_effort_raw = np.linalg.norm(action)
    cost_effort = rew_coeff["effort"] * cost_effort_raw

    # Loss orientation
    if on_floor:
        cost_orient_raw = 1.0
    else:
        cost_orient_raw = -dynamics.rot[2, 2]

    cost_orient = rew_coeff["orient"] * cost_orient_raw

    # Loss crash for staying on the floor
    cost_crash_raw = float(on_floor)
    cost_crash = rew_coeff["crash"] * cost_crash_raw

    # Loss for constant uncontrolled rotation around vertical axis
    cost_spin_raw = (dynamics.omega[0] ** 2 + dynamics.omega[1] ** 2 + dynamics.omega[2] ** 2) ** 0.5
    cost_spin = rew_coeff["spin"] * cost_spin_raw

    if log_all_info:
        # Projection of the z-body axis to z-world axis
        # Negative, because the larger the projection the smaller the loss (i.e. the higher the reward)
        rot_cos = ((dynamics.rot[0, 0] + dynamics.rot[1, 1] + dynamics.rot[2, 2]) - 1.) / 2.
        # We have to clip since rotation matrix falls out of orthogonalization from time to time
        cost_rotation_raw = np.arccos(np.clip(rot_cos, -1., 1.))  # angle = arccos((trR-1)/2) See: [6]
        cost_rotation = rew_coeff["rot"] * cost_rotation_raw

        cost_attitude_raw = np.arccos(np.clip(dynamics.rot[2, 2], -1., 1.))
        cost_attitude = rew_coeff["attitude"] * cost_attitude_raw

        cost_yaw_raw = -dynamics.rot[0, 0]
        cost_yaw = rew_coeff["yaw"] * cost_yaw_raw

        dact = action - action_prev
        cost_act_change_raw = (dact[0] ** 2 + dact[1] ** 2 + dact[2] ** 2 + dact[3] ** 2) ** 0.5
        cost_act_change = rew_coeff["action_change"] * cost_act_change_raw

        # loss velocity
        cost_vel_raw = np.linalg.norm(dynamics.vel)
        cost_vel = rew_coeff["vel"] * cost_vel_raw

    reward = -dt * np.sum([
        cost_pos,
        cost_effort,
        cost_crash,
        cost_orient,
        cost_spin,
        # cost_yaw,
        # cost_rotation,
        # cost_attitude,
        # cost_act_change,
        # cost_vel,
    ])

    rew_info = {
        "rew_main": -cost_pos,
        'rew_pos': -cost_pos,
        'rew_action': -cost_effort,
        'rew_crash': -cost_crash,
        "rew_orient": -cost_orient,
        "rew_spin": -cost_spin,
        # "rew_yaw": -cost_yaw,
        # "rew_rot": -cost_rotation,
        # "rew_attitude": -cost_attitude,
        # "rew_act_change": -cost_act_change,
        # "rew_vel": -cost_vel,

        "rewraw_main": -cost_pos_raw,
        'rewraw_pos': -cost_pos_raw,
        'rewraw_action': -cost_effort_raw,
        'rewraw_crash': -cost_crash_raw,
        "rewraw_orient": -cost_orient_raw,
        "rewraw_spin": -cost_spin_raw,
        # "rewraw_yaw": -cost_yaw_raw,
        # "rewraw_rot": -cost_rotation_raw,
        # "rewraw_attitude": -cost_attitude_raw,
        # "rewraw_act_change": -cost_act_change_raw,
        # "rewraw_vel": -cost_vel_raw,
    }

    # report rewards in the same format as they are added to the actual agent's reward (easier to debug this way)
    for k, v in rew_info.items():
        rew_info[k] = dt * v

    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info


# Gym environment for quadrotor seeking the origin with no obstacles and full state observations.
# NOTES:
# - room size of the env and init state distribution are not the same !
#   It is done for the reason of having static (and preferably short) episode length, since for some distance
#   it would be impossible to reach the goal
class QuadrotorSingle:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dynamics_params="DefaultQuad", dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr="xyz_vxyz_R_omega", ep_time=7, room_dims=(10.0, 10.0, 10.0),
                 init_random_state=False, rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False, use_numba=False,
                 neighbor_obs_type='none', num_agents=1,
                 view_mode='local', num_use_neighbor_obs=0, use_obstacles=False, obst_obs_type='none'):
        np.seterr(under='ignore')
        """
        Args:
            dynamics_params: [str or dict] loading dynamics params by name or by providing a dictionary. 
                If "random": dynamics will be randomized completely (see sample_dyn_parameters() )
                If dynamics_randomize_every is None: it will be randomized only once at the beginning.
                One can randomize dynamics during the end of any episode using resample_dynamics()
                WARNING: randomization during an episode is not supported yet. Randomize ONLY before calling reset().
            dynamics_change: [dict] update to dynamics parameters relative to dynamics_params provided
            
            dynamics_randomize_every: [int] how often (trajectories) perform randomization
            dynamics_sampler_1: [dict] the first sampler to be applied. Dict must contain type 
                (see quadrotor_randomization) and whatever params samler requires
            dynamics_sampler_2: [dict] the second sampler to be applied. Convenient if you need to 
            fix some params after sampling.
            
            raw_control: [bool] use raw cantrol or the Mellinger controller as a default
            raw_control_zero_middle: [bool] meaning that control will be [-1 .. 1] rather than [0 .. 1]
            dim_mode: [str] Dimensionality of the env. Options: 
                1D(just a vertical stabilization), 2D(vertical plane), 3D(normal)
            tf_control: [bool] creates Mellinger controller using TensorFlow
            sim_freq (float): frequency of simulation
            sim_steps: [int] how many simulation steps for each control step
            obs_repr: [str] options: xyz_vxyz_rot_omega, xyz_vxyz_quat_omega
            ep_time: [float] episode time in simulated seconds. 
                This parameter is used to compute env max time length in steps.
            init_random_state: [bool] use random state initialization or horizontal initialization with 0 velocities
            rew_coeff: [dict] weights for different reward components (see compute_weighted_reward() function)
            sens_noise (dict or str): sensor noise parameters. If None - no noise. 
                If "default" then the default params are loaded. Otherwise one can provide specific params.
            excite: [bool] change the setpoint at the fixed frequency to perturb the quad
        """
        # Params
        self.gravity = gravity
        # t2w and t2t ranges
        self.t2w_std = t2w_std
        self.t2w_min = 1.5
        self.t2w_max = 10.0

        self.t2t_std = t2t_std
        self.t2t_min = 0.005
        self.t2t_max = 1.0
        self.excite = excite

        self.max_init_vel = 1.  # m/s
        self.max_init_omega = 2 * np.pi  # rad/s

        self.traj_count = 0

        # # Print
        self.verbose = verbose

        # # EPISODE PARAMS
        self.sim_steps = sim_steps
        self.ep_time = ep_time  # In seconds
        self.dt = 1.0 / sim_freq
        self.metadata["video.frames_per_second"] = int(sim_freq / self.sim_steps)
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        self.tick = 0
        self.crashed = False
        self.control_freq = sim_freq / sim_steps
        self.rew_coeff = rew_coeff

        # Dynamics
        self.dynamics = None
        self.controller = None
        self.init_random_state = init_random_state
        self.dim_mode = dim_mode
        self.raw_control_zero_middle = raw_control_zero_middle
        self.tf_control = tf_control
        self.dynamics_randomize_every = dynamics_randomize_every
        self.raw_control = raw_control
        self.dynamics_simplification = dynamics_simplification

        # Could be dynamics of a specific quad or a random dynamics (i.e. randomquad)
        self.dyn_base_sampler = getattr(quad_rand, dynamics_params)()
        self.dynamics_change = copy.deepcopy(dynamics_change)

        self.dynamics_params = self.dyn_base_sampler.sample()
        # Now, updating if we are providing modifications
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        self.dyn_sampler_1 = dyn_sampler_1
        if dyn_sampler_1 is not None:
            sampler_type = dyn_sampler_1["class"]
            self.dyn_sampler_1_params = copy.deepcopy(dyn_sampler_1)
            del self.dyn_sampler_1_params["class"]
            self.dyn_sampler_1 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_1_params)

        self.dyn_sampler_2 = dyn_sampler_2
        if dyn_sampler_2 is not None:
            sampler_type = dyn_sampler_2["class"]
            self.dyn_sampler_2_params = copy.deepcopy(dyn_sampler_2)
            del self.dyn_sampler_2_params["class"]
            self.dyn_sampler_2 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_2_params)

        # Updating dynamics
        dyn_upd_start_time = time.time()
        # Also performs update of the dynamics
        self.action_space = None  # to be defined in update_dynamics
        self.resample_dynamics()
        # self.update_dynamics(dynamics_params=self.dynamics_params)
        print("QuadEnv: Dyn update time: ", time.time() - dyn_upd_start_time)

        # Obs
        self.obs_repr = obs_repr
        self.state_vector = self.state_vector = getattr(get_state, "state_" + self.obs_repr)
        # # Sense Noise
        self.sense_noise = None
        self.update_sense_noise(sense_noise=sense_noise)

        self.observation_space = self.make_observation_space()
        self.obst_obs_type = obst_obs_type
        self.obs_space_low_high = None

        # Neighbor
        self.num_agents = num_agents
        self.num_use_neighbor_obs = num_use_neighbor_obs
        self.neighbor_obs_type = neighbor_obs_type

        # Obstacles
        self.use_obstacles = use_obstacles

        # Room
        self.room_length = room_dims[0]
        self.room_width = room_dims[1]
        self.room_height = room_dims[2]
        self.room_box = np.array(
            [[-self.room_length / 2., -self.room_width / 2, 0.],
             [self.room_length / 2., self.room_width / 2., self.room_height]])

        # Numba Speed Up
        self.use_numba = use_numba

        # Goal
        self.goal = None

        # Spawn around goal box
        if self.use_obstacles:
            self.box = 0.1
        else:
            self.box = 2.0
        self.box_scale = 1.0  # scale the initialbox by this factor eache episode

        # Rendering
        self.view_mode = view_mode

        # Seed
        self._seed()

    def update_sense_noise(self, sense_noise):
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False, use_numba=self.use_numba)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    def update_dynamics(self, dynamics_params):
        # DYNAMICS
        # # Then loading the dynamics
        self.dynamics_params = dynamics_params
        self.dynamics = QuadrotorDynamics(model_params=dynamics_params,
                                          dynamics_steps_num=self.sim_steps, dt=self.dt, room_box=self.room_box,
                                          dim_mode=self.dim_mode,
                                          gravity=self.gravity, dynamics_simplification=self.dynamics_simplification,
                                          use_numba=self.use_numba)

        if self.verbose:
            print("#################################################")
            print("Dynamics params loaded:")
            print_dic(dynamics_params)
            print("#################################################")

        # # CONTROL
        if self.raw_control:
            if self.dim_mode == '1D':  # Z axis only
                self.controller = VerticalControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '2D':  # X and Z axes only
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            else:
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        else:
            self.controller = NonlinearPositionController(self.dynamics, tf_control=self.tf_control)

        # # ACTIONS
        self.action_space = self.controller.action_space(self.dynamics)

        # # STATE VECTOR FUNCTION
        self.state_vector = getattr(get_state, "state_" + self.obs_repr)

    def make_observation_space(self):
        room_range = self.room_box[1] - self.room_box[0]
        self.obs_space_low_high = {
            "xyz": [-room_range, room_range],
            "xyzr": [-room_range, room_range],
            "vxyz": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "vxyzr": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "acc": [-self.dynamics.acc_max * np.ones(3), self.dynamics.acc_max * np.ones(3)],
            "R": [-np.ones(9), np.ones(9)],
            "omega": [-self.dynamics.omega_max * np.ones(3), self.dynamics.omega_max * np.ones(3)],
            "t2w": [0. * np.ones(1), 5. * np.ones(1)],
            "t2t": [0. * np.ones(1), 1. * np.ones(1)],
            "h": [0. * np.ones(1), self.room_box[1][2] * np.ones(1)],
            "act": [np.zeros(4), np.ones(4)],
            "quat": [-np.ones(4), np.ones(4)],
            "euler": [-np.pi * np.ones(3), np.pi * np.ones(3)],
            "rxyz": [-room_range, room_range],  # rxyz stands for relative pos between quadrotors
            "rvxyz": [-2.0 * self.dynamics.vxyz_max * np.ones(3), 2.0 * self.dynamics.vxyz_max * np.ones(3)],
            # rvxyz stands for relative velocity between quadrotors
            "roxyz": [-room_range, room_range],  # roxyz stands for relative pos between quadrotor and obstacle
            "rovxyz": [-20.0 * np.ones(3), 20.0 * np.ones(3)],
            # rovxyz stands for relative velocity between quadrotor and obstacle
            "osize": [np.zeros(3), 20.0 * np.ones(3)],  # obstacle size, [[0., 0., 0.], [20., 20., 20.]]
            # obstacle type, [[0.], [20.]], which means we can support 21 types of obstacles
            "wall": [np.zeros(6), 5.0 * np.ones(6)],
            "floor": [0. * np.ones(1), self.room_box[1][2] * np.ones(1)],
            "octomap_2d": [-10 * np.ones(9), 10 * np.ones(9)],
        }

        obs_comps = self.obs_repr.split("_")
        if self.neighbor_obs_type == 'pos_vel' and self.num_use_neighbor_obs > 0:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz']) * self.num_use_neighbor_obs
        if self.use_obstacles:
            if self.obst_obs_type == "octomap_2d":
                obs_comps = obs_comps + ["octomap_2d"]

        print("Observation components:", obs_comps)
        obs_low, obs_high = [], []
        for comp in obs_comps:
            obs_low.append(self.obs_space_low_high[comp][0])
            obs_high.append(self.obs_space_low_high[comp][1])
        obs_low = np.concatenate(obs_low)
        obs_high = np.concatenate(obs_high)

        observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return observation_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.actions[1] = copy.deepcopy(self.actions[0])
        self.actions[0] = copy.deepcopy(action)

        self.controller.step_func(dynamics=self.dynamics,
                                  action=action,
                                  goal=self.goal,
                                  dt=self.dt,
                                  observation=None)

        self.time_remain = self.ep_len - self.tick

        # dynamics, goal, action, dt, time_remain, rew_coeff, action_prev, on_floor = False
        reward, rew_info = compute_reward_weighted(dynamics=self.dynamics, goal=self.goal, action=action, dt=self.dt,
                                                   time_remain=self.time_remain, rew_coeff=self.rew_coeff,
                                                   action_prev=self.actions[1], on_floor=self.dynamics.on_floor)

        self.tick += 1
        done = self.tick > self.ep_len
        self.traj_count += int(done)

        # Self Obs
        sv = self.state_vector(self)

        return sv, reward, done, {'rewards': rew_info}

    def resample_dynamics(self):
        """
        Allows manual dynamics resampling when needed.
        WARNING: 
            - Randomization dyring an episode is not supported
            - MUST call reset() after this function
        """
        # Getting base parameters (could also be random parameters)
        self.dynamics_params = self.dyn_base_sampler.sample()

        # Now, updating if we are providing modifications
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        # Applying sampler 1
        if self.dyn_sampler_1 is not None:
            self.dynamics_params = self.dyn_sampler_1.sample(self.dynamics_params)

        # Applying sampler 2
        if self.dyn_sampler_2 is not None:
            self.dynamics_params = self.dyn_sampler_2.sample(self.dynamics_params)

        # Checking that quad params make sense
        quad_rand.check_quad_param_limits(self.dynamics_params)

        # Updating params
        self.update_dynamics(dynamics_params=self.dynamics_params)

    def _reset(self):
        # Have to update state vector
        # DYNAMICS RANDOMIZATION AND UPDATE
        if self.dynamics_randomize_every is not None and (self.traj_count + 1) % self.dynamics_randomize_every == 0:
            self.resample_dynamics()

        if self.box < 10:
            self.box = self.box * self.box_scale
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.goal

        if self.dim_mode == '1D':
            x, y = self.goal[0], self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]
        # Since being near the groud means crash we have to start above
        if z < 0.25:
            z = 0.25

        pos = npa(x, y, z)

        # INIT STATE
        # Initializing rotation and velocities
        if self.init_random_state:
            if self.dim_mode == '1D':
                omega, rotation = np.zeros(3, dtype=np.float64), np.eye(3)
                vel = np.array([0, 0, self.max_init_vel * np.random.rand()])
            elif self.dim_mode == '2D':
                omega = npa(0, self.max_init_omega * np.random.rand(), 0)
                vel = self.max_init_vel * np.random.rand(3)
                vel[1] = 0.
                theta = np.pi * np.random.rand()
                c, s = np.cos(theta), np.sin(theta)
                rotation = np.array(((c, 0, -s), (0, 1, 0), (s, 0, c)))
            else:
                # It already sets the state internally
                _, vel, rotation, omega = self.dynamics.random_state(
                    box=(self.room_length, self.room_width, self.room_height), vel_max=self.max_init_vel,
                    omega_max=self.max_init_omega
                )
        else:
            # INIT HORIZONTALLY WITH 0 VEL and OMEGA
            vel, omega = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

            if self.dim_mode == '1D' or self.dim_mode == '2D':
                rotation = np.eye(3)
            else:
                # make sure we're sort of pointing towards goal (for mellinger controller)
                rotation = randyaw()
                while np.dot(rotation[:, 0], to_xyhat(-pos)) < 0.5:
                    rotation = randyaw()

        # Setting the generated state
        self.init_state = [pos, vel, rotation, omega]
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.dynamics.reset()
        self.dynamics.on_floor = False
        self.dynamics.crashed_floor = self.dynamics.crashed_wall = self.dynamics.crashed_ceiling = False

        # Resetting some internal state (counters, etc)
        self.crashed = False
        self.tick = 0
        self.actions = [np.zeros([4, ]), np.zeros([4, ])]

        state = self.state_vector(self)
        return state

    def reset(self):
        return self._reset()

    def render(self, mode='human', **kwargs):
        """This class is only meant to be used as a component of QuadMultiEnv."""
        raise NotImplementedError()

    def step(self, action):
        return self._step(action)


class DummyPolicy(object):
    def __init__(self, dt=0.01, switch_time=2.5):
        self.action = np.zeros([4, ])
        self.dt = 0.

    def step(self, x):
        return self.action

    def reset(self):
        pass


class UpDownPolicy(object):
    def __init__(self, dt=0.01, switch_time=2.5):
        self.t = 0
        self.dt = dt
        self.switch_time = switch_time
        self.action_up = np.ones([4, ])
        self.action_up[:2] = 0.
        self.action_down = np.zeros([4, ])
        self.action_down[:2] = 1.

    def step(self, x):
        self.t += self.dt
        if self.t < self.switch_time:
            return self.action_up
        else:
            return self.action_down

    def reset(self):
        self.t = 0.


@njit
def calculate_torque_integrate_rotations_and_update_omega(
        thrust_cmds, dt, motor_tau_up, motor_tau_down, thrust_cmds_damp, thrust_rot_damp, thr_noise,
        thrust_max, motor_linearity, prop_crossproducts, prop_ccw, torque_max, rot, omega, eye, since_last_svd,
        since_last_svd_limit, inertia, damp_omega_quadratic, omega_max, pos, vel):
    # Filtering the thruster and adding noise
    thrust_cmds = np.clip(thrust_cmds, 0., 1.)
    motor_tau = motor_tau_up * np.ones(4)
    motor_tau[thrust_cmds < thrust_cmds_damp] = np.array(motor_tau_down)
    motor_tau[motor_tau > 1.] = 1.

    # Since NN commands thrusts we need to convert to rot vel and back
    thrust_rot = thrust_cmds ** 0.5
    thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp
    thrust_cmds_damp = thrust_rot_damp ** 2

    # Adding noise
    thrust_noise = thrust_cmds * thr_noise
    thrust_cmds_damp = np.clip(thrust_cmds_damp + thrust_noise, 0.0, 1.0)
    thrusts = thrust_max * angvel2thrust_numba(thrust_cmds_damp, motor_linearity)

    # Prop cross-product gives torque directions
    torques = prop_crossproducts * np.reshape(thrusts, (-1, 1))

    # Additional torques along z-axis caused by propeller rotations
    torques[:, 2] += torque_max * prop_ccw * thrust_cmds_damp

    # Net torque: sum over propellers
    thrust_torque = np.sum(torques, 0)

    # Rotor drag and Rolling forces and moments
    rotor_visc_torque = rotor_drag_force = np.zeros(3)

    # (Square) Damping using torques (in case we would like to add damping using torques)
    torque = thrust_torque + rotor_visc_torque
    thrust = np.array([0., 0., np.sum(thrusts)])

    # ROTATIONAL DYNAMICS
    # Integrating rotations (based on current values)
    omega_vec = rot @ omega
    wx, wy, wz = omega_vec
    omega_norm = np.linalg.norm(omega_vec)
    if omega_norm != 0:
        K = np.array([[0., -wz, wy], [wz, 0., -wx], [-wy, wx, 0.]]) / omega_norm
        rot_angle = omega_norm * dt
        dRdt = eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
        rot = dRdt @ rot

    # SVD is not strictly required anymore. Performing it rarely, just in case
    since_last_svd += dt
    if since_last_svd > since_last_svd_limit:
        u, s, v = np.linalg.svd(rot)
        rot = u @ v
        since_last_svd = 0

    # COMPUTING OMEGA UPDATE
    # Linear damping
    omega_dot = ((1.0 / inertia) * (numba_cross(-omega, inertia * omega) + torque))

    # Quadratic damping
    omega_damp_quadratic = np.clip(damp_omega_quadratic * omega ** 2, 0.0, 1.0)
    omega = omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
    omega = np.clip(omega, -omega_max, omega_max)

    # Computing position
    pos = pos + dt * vel

    return thrust_rot_damp, thrust_cmds_damp, rot, since_last_svd, omega, pos, thrust, rotor_drag_force, vel


@njit
def compute_velocity_and_acceleration(vel, vel_damp, dt, rot_tpose, grav_arr, acc):
    # Computing velocities
    vel = (1.0 - vel_damp) * vel + dt * acc

    # Accelerometer measures so called "proper acceleration" that includes gravity with the opposite sign
    accm = rot_tpose @ (acc + grav_arr)
    return vel, accm


@njit
def floor_interaction_numba(pos, vel, rot, omega, mu, mass, sum_thr_drag, thrust_cmds_damp, thrust_rot_damp,
                            floor_threshold, on_floor):
    # Change pos, omega, rot, acc
    crashed_floor = False
    if pos[2] <= floor_threshold:
        pos = np.array((pos[0], pos[1], floor_threshold))
        force = rot @ sum_thr_drag
        if on_floor:
            # Drone is on the floor, and on_floor flag still True
            theta = np.arctan2(rot[1][0], rot[0][0] + EPS)
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

            # Add friction if drone is on the floor
            f = mu * GRAV * np.array((np.sign(force[0]), np.sign(force[1]), 0)) * mass
            # Since fiction cannot be greater than force, we need to clip it
            for i in range(2):
                if np.abs(f[i]) > np.abs(force[i]):
                    f[i] = force[i]
            force -= f

        else:
            # Previous step, drone still in the air, but in this step, it hits the floor
            # In previous step, self.on_floor = False, self.crashed_floor = False
            on_floor = True
            crashed_floor = True
            # Set vel to [0, 0, 0]
            vel, acc = np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            omega = np.zeros(3, dtype=np.float64)
            # Set rot
            theta = np.arctan2(rot[1][0], rot[0][0] + EPS)
            c, s = np.cos(theta), np.sin(theta)
            if rot[2, 2] < 0:
                theta = np.random.uniform(-np.pi, np.pi)
                c, s = np.cos(theta), np.sin(theta)
                rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            else:
                rot = np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))

            # reset momentum / accumulation of thrust
            thrust_cmds_damp = np.zeros(4)
            thrust_rot_damp = np.zeros(4)

        acc = np.array((0., 0., -GRAV)) + (1.0 / mass) * force
        acc[2] = np.maximum(0, acc[2])
    else:
        # self.pos[2] > self.floor_threshold
        if on_floor:
            # Drone is in the air, while on_floor flag still True
            on_floor = False

        # Computing accelerations
        force = rot @ sum_thr_drag
        acc = np.array((0., 0., -GRAV)) + (1.0 / mass) * force

    return pos, vel, acc, omega, rot, thrust_cmds_damp, thrust_rot_damp, on_floor, crashed_floor
