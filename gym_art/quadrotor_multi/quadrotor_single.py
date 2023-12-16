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
from collections import deque

import numpy as np
from gymnasium.utils import seeding

import gym_art.quadrotor_multi.get_state as get_state
import gym_art.quadrotor_multi.quadrotor_randomization as quad_rand
from gym_art.quadrotor_multi.quadrotor_control import *
from gym_art.quadrotor_multi.quadrotor_dynamics import QuadrotorDynamics
from gym_art.quadrotor_multi.sensor_noise import SensorNoise

GRAV = 9.81  # default gravitational constant


# reasonable reward function for hovering at a goal and not flying too high
def compute_reward_weighted(
        goal, cur_pos, rl_acc, acc_sbc, sbc_distance_to_boundary, mellinger_acc, dt, rew_coeff, on_floor, action_prev,
        aggressive_unclip, enable_sbc):
    # Distance to the goal
    dist = np.linalg.norm(goal - cur_pos)
    cost_pos_raw = dist
    cost_pos = rew_coeff["pos"] * cost_pos_raw

    # Loss crash for staying on the floor
    cost_crash_raw = float(on_floor)
    cost_crash = rew_coeff["crash"] * cost_crash_raw

    # Difference between acc_rl & acc_sbc
    if enable_sbc:
        cost_rl_sbc_raw = np.linalg.norm(acc_sbc - rl_acc)
        cost_rl_sbc = rew_coeff["rl_sbc"] * cost_rl_sbc_raw
    else:
        cost_rl_sbc_raw = 0.0
        cost_rl_sbc = 0.0

    # Difference between acc_rl & acc_mellinger
    cost_rl_mellinger_raw = np.linalg.norm(mellinger_acc - rl_acc)
    cost_rl_mellinger = rew_coeff["rl_mellinger"] * cost_rl_mellinger_raw

    # Action change
    cost_act_change_raw = np.linalg.norm(action_prev - rl_acc)
    cost_act_change = rew_coeff["act_change"] * cost_act_change_raw

    # Aggressiveness unclip
    if enable_sbc:
        aggressiveness_clip = np.clip(aggressive_unclip, a_min=0.0, a_max=1.0)
        agg_clipped_list = np.abs(aggressive_unclip - aggressiveness_clip)
        agg_clipped_cost = agg_clipped_list[0] + agg_clipped_list[1]
        agg_too_large_cost = aggressiveness_clip[0] + aggressiveness_clip[1]

        cost_aggressiveness_raw = agg_clipped_cost + agg_too_large_cost
        cost_aggressiveness = rew_coeff["cbg_agg"] * cost_aggressiveness_raw
    else:
        cost_aggressiveness_raw = 0.0
        cost_aggressiveness = 0.0

    # SBC boundary cost
    if enable_sbc:
        if sbc_distance_to_boundary is not None:
            cost_sbc_boundary_raw = sbc_distance_to_boundary
            cost_sbc_boundary = rew_coeff["sbc_boundary"] * cost_sbc_boundary_raw
        else:
            cost_sbc_boundary_raw = 0.
            cost_sbc_boundary = 0.
    else:
        cost_sbc_boundary_raw = 0.
        cost_sbc_boundary = 0.

    reward = -dt * np.sum([
        cost_pos,
        cost_crash,
        cost_rl_sbc,
        cost_rl_mellinger,
        cost_sbc_boundary,
        cost_act_change,
        cost_aggressiveness,
    ])

    rew_info = {
        "rew_main": -cost_pos,
        'rew_pos': -cost_pos,
        'rew_crash': -cost_crash,
        "rew_rl_sbc": -cost_rl_sbc,
        'rew_rl_mellinger': -cost_rl_mellinger,
        'rew_sbc_boundary': -cost_sbc_boundary,
        'rew_act_change': -cost_act_change,
        'rew_cbf_agg': -cost_aggressiveness,

        "rewraw_main": -cost_pos_raw,
        'rewraw_pos': -cost_pos_raw,
        'rewraw_crash': -cost_crash_raw,
        "rewraw_rl_sbc": -cost_rl_sbc_raw,
        'rewraw_rl_mellinger': -cost_rl_mellinger_raw,
        'rewraw_sbc_boundary': -cost_sbc_boundary_raw,
        'rewraw_act_change': -cost_act_change_raw,
        'rewraw_cbf_agg': -cost_aggressiveness_raw,
    }

    for k, v in rew_info.items():
        rew_info[k] = dt * v

    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info


# ENV Gym environment for quadrotor seeking the origin with no obstacles and full state observations. NOTES: - room
# size of the env and init state distribution are not the same ! It is done for the reason of having static (and
# preferably short) episode length, since for some distance it would be impossible to reach the goal
class QuadrotorSingle:
    def __init__(
            # self
            self, dynamics_params="DefaultQuad", dynamics_change=None, dynamics_randomize_every=None,
            dyn_sampler_1=None, dyn_sampler_2=None, raw_control=True, raw_control_zero_middle=True, sense_noise=None,
            init_random_state=False, obs_repr="xyz_vxyz_R_omega", ep_time=7, room_dims=(10.0, 10.0, 10.0),
            use_numba=False, his_acc=False, his_acc_num=3, obs_single_his_acc=11,
            # Neighbor
            num_agents=1, neighbor_obs_type='none', num_use_neighbor_obs=0,
            # Obstacle
            use_obstacles=False, num_obstacles=0,
            # SBC specific,
            enable_sbc=True, sbc_radius=0.1, sbc_max_acc=2.0,
            # Others
            dim_mode='3D', tf_control=False, sim_freq=200., sim_steps=2, verbose=False, gravity=GRAV, t2w_std=0.005,
            t2t_std=0.0005, excite=False, dynamics_simplification=False):
        np.seterr(under='ignore')
        """
        Args:
            dynamics_params: [str or dict] loading dynamics params by name or by providing a dictionary. 
                If "random": dynamics will be randomized completely (see sample_dyn_parameters() )
                If dynamics_randomize_every is None: it will be randomized only once at the beginning.
                One can randomize dynamics during the end of any episode using resample_dynamics()
                WARNING: randomization during an episode is not supported yet. Randomize ONLY before calling reset().
            dynamics_change: [dict] update to dynamics parameters relative to dynamics_params provided
            
            dynamics_randomize_every: [int] how often (trajectories) perform randomization dynamics_sampler_1: [dict] 
            the first sampler to be applied. Dict must contain type (see quadrotor_randomization) and whatever params 
            requires 
            dynamics_sampler_2: [dict] the second sampler to be applied. Convenient if you need to 
                fix some params after sampling.
            
            raw_control: [bool] use raw control or the Mellinger controller as a default
            raw_control_zero_middle: [bool] meaning that control will be [-1 .. 1] rather than [0 .. 1]
            dim_mode: [str] Dimensionality of the env. 
            Options: 1D(just a vertical stabilization), 2D(vertical plane), 3D(normal)
            tf_control: [bool] creates Mellinger controller using TensorFlow
            sim_freq (float): frequency of simulation
            sim_steps: [int] how many simulation steps for each control step
            obs_repr: [str] options: xyz_vxyz_rot_omega, xyz_vxyz_quat_omega
            ep_time: [float] episode time in simulated seconds. 
                This parameter is used to compute env max time length in steps.
            room_size: [int] env room size. Not the same as the initialization box to allow shorter episodes
            init_random_state: [bool] use random state initialization or horizontal initialization with 0 velocities
            rew_coeff: [dict] weights for different reward components (see compute_weighted_reward() function)
            sens_noise (dict or str): sensor noise parameters. If None - no noise. If "default" then the default params 
                are loaded. Otherwise one can provide specific params.
            excite: [bool] change the set point at the fixed frequency to perturb the quad
        """
        # Numba Speed Up
        self.use_numba = use_numba
        # Neighbor info
        self.num_agents = num_agents
        self.neighbor_obs_type = neighbor_obs_type
        self.num_use_neighbor_obs = num_use_neighbor_obs
        # Obstacles info
        self.use_obstacles = use_obstacles
        self.num_obstacles = num_obstacles

        # # SBC specific
        self.enable_sbc = enable_sbc
        if enable_sbc:
            self.sbc_radius = sbc_radius
            self.sbc_max_acc = sbc_max_acc
        else:
            self.sbc_radius = 0.05
            self.sbc_max_acc = 2.0

        # Room
        self.room_length = room_dims[0]
        self.room_width = room_dims[1]
        self.room_height = room_dims[2]
        self.room_box = np.array(
            [[-self.room_length / 2., -self.room_width / 2, 0.],
             [self.room_length / 2., self.room_width / 2., self.room_height]])

        self.init_random_state = init_random_state

        # Preset parameters
        self.obs_repr = obs_repr
        self.actions = [np.zeros([3, ]), np.zeros([3, ])]
        self.rew_coeff = None
        self.his_acc = his_acc
        self.his_acc_num = his_acc_num
        if his_acc:
            # acc_ref, acc_sbc, acc_real, aggressiveness for neighbor, aggressiveness for obstacle
            self.obs_single_his_acc = obs_single_his_acc
            self.obs_his_accs = deque([], maxlen=his_acc_num)
        # EPISODE PARAMS
        self.ep_time = ep_time  # In seconds
        self.sim_steps = sim_steps
        self.dt = 1.0 / sim_freq
        self.time_remain = 0
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        self.tick = 0
        self.control_freq = sim_freq / sim_steps
        self.control_dt = 1.0 / self.control_freq
        self.traj_count = 0

        # Self dynamics
        self.dim_mode = dim_mode
        self.raw_control_zero_middle = raw_control_zero_middle
        self.tf_control = tf_control
        self.dynamics_randomize_every = dynamics_randomize_every
        self.verbose = verbose
        self.raw_control = raw_control
        self.gravity = gravity
        self.update_sense_noise(sense_noise=sense_noise)
        self.t2w_std = t2w_std
        self.t2w_min = 1.5
        self.t2w_max = 10.0

        self.t2t_std = t2t_std
        self.t2t_min = 0.005
        self.t2t_max = 1.0
        self.excite = excite
        self.dynamics_simplification = dynamics_simplification
        self.max_init_vel = 1.  # m/s
        self.max_init_omega = 2 * np.pi  # rad/s

        # DYNAMICS (and randomization)
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
            self.dyn_sampler_1 = getattr(quad_rand, sampler_type)(
                params=self.dynamics_params, **self.dyn_sampler_1_params)

        self.dyn_sampler_2 = dyn_sampler_2
        if dyn_sampler_2 is not None:
            sampler_type = dyn_sampler_2["class"]
            self.dyn_sampler_2_params = copy.deepcopy(dyn_sampler_2)
            del self.dyn_sampler_2_params["class"]
            self.dyn_sampler_2 = getattr(quad_rand, sampler_type)(
                params=self.dynamics_params, **self.dyn_sampler_2_params)

        # Updating dynamics
        self.action_space = None
        self.resample_dynamics()

        # Self info
        self.state_vector = self.state_vector = getattr(
            get_state, "state_" + self.obs_repr)
        if use_obstacles:
            self.box = 0.1
        else:
            self.box = 2.0
        self.box_scale = 1.0
        self.goal = None
        self.spawn_point = None

        # Make observation space
        self.observation_space = self.make_observation_space()

        self._seed()

    def update_sense_noise(self, sense_noise):
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(
                    bypass=False, use_numba=self.use_numba)
            else:
                ValueError(
                    "ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError(
                "ERROR: QuadEnv: sense_noise parameter is of unknown type: " +
                str(sense_noise))

    def update_dynamics(self, dynamics_params):
        # DYNAMICS
        # Then loading the dynamics
        self.dynamics_params = dynamics_params
        self.dynamics = QuadrotorDynamics(
            model_params=dynamics_params, dynamics_steps_num=self.sim_steps,
            room_box=self.room_box, dim_mode=self.dim_mode, gravity=self.
            gravity, dynamics_simplification=self.dynamics_simplification,
            use_numba=self.use_numba, dt=self.dt)

        # CONTROL
        self.controller = MellingerController(
            dynamics=self.dynamics, sbc_radius=self.sbc_radius,
            room_box=self.room_box, num_agents=self.num_agents, num_obstacles=self.num_obstacles,
            sbc_max_acc=self.sbc_max_acc, enable_sbc=self.enable_sbc)

        # ACTIONS
        if self.enable_sbc:
            action_lows_space = np.array([-1, -1, -1, 0, 0], dtype=np.float32)
            action_high_space = np.array([1, 1, 1, 1, 1], dtype=np.float32)
        else:
            action_lows_space = np.array([-1, -1, -1], dtype=np.float32)
            action_high_space = np.array([1, 1, 1], dtype=np.float32)

        self.action_space = spaces.Box(low=action_lows_space, high=action_high_space, dtype=np.float32)

        # STATE VECTOR FUNCTION
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
            # rxyz stands for relative pos between quadrotors
            "rxyz": [-room_range, room_range],
            "rvxyz": [-2.0 * self.dynamics.vxyz_max * np.ones(3), 2.0 * self.dynamics.vxyz_max * np.ones(3)],
            # rvxyz stands for relative velocity between quadrotors
            # roxyz stands for relative pos between quadrotor and obstacle
            "roxyz": [-room_range, room_range],
            "rovxyz": [-20.0 * np.ones(3), 20.0 * np.ones(3)],
            # rovxyz stands for relative velocity between quadrotor and obstacle size, [[0., 0., 0.], [20., 20., 20.]]
            "osize": [np.zeros(3), 20.0 * np.ones(3)],
            "otype": [np.zeros(1), 20.0 * np.ones(1)],
            # obstacle type, [[0.], [20.]], which means we can support 21 types of obstacles
            "goal": [-room_range, room_range],
            "wall": [np.zeros(6), 5.0 * np.ones(6)],
            "floor": [np.zeros(1), self.room_box[1][2] * np.ones(1)],
            "octmap": [-10 * np.ones(9), 10 * np.ones(9)],
            # CBF, aggressiveness
            'agg': [np.zeros(1), 100. * np.ones(1)]
        }
        self.obs_comp_names = list(self.obs_space_low_high.keys())
        self.obs_comp_sizes = [self.obs_space_low_high[name][1].size for name in self.obs_comp_names]

        obs_comps = self.obs_repr.split("_")
        if self.his_acc:
            if self.enable_sbc:
                obs_comps = obs_comps + (['acc'] * 3 + ['agg'] * 2) * self.his_acc_num
            else:
                obs_comps = obs_comps + (['acc'] * 2) * self.his_acc_num

        if self.neighbor_obs_type == 'pos_vel' and self.num_use_neighbor_obs > 0:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz']) * self.num_use_neighbor_obs

        if self.use_obstacles:
            obs_comps = obs_comps + ["octmap"]

        print("Observation components:", obs_comps)
        obs_low, obs_high = [], []
        for comp in obs_comps:
            obs_low.append(self.obs_space_low_high[comp][0])
            obs_high.append(self.obs_space_low_high[comp][1])
        obs_low = np.concatenate(obs_low)
        obs_high = np.concatenate(obs_high)

        self.obs_comp_sizes_dict, self.obs_space_comp_indx, self.obs_comp_end = {}, {}, []
        end_indx = 0
        for obs_i, obs_name in enumerate(self.obs_comp_names):
            end_indx += self.obs_comp_sizes[obs_i]
            self.obs_comp_sizes_dict[obs_name] = self.obs_comp_sizes[obs_i]
            self.obs_space_comp_indx[obs_name] = obs_i
            self.obs_comp_end.append(end_indx)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return self.observation_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action, sbc_data):
        # since action means action change
        # acc = pre_acc + action
        # desired_acc = action + self.dynamics.acc

        self.actions[1] = copy.deepcopy(self.actions[0])
        self.actions[0] = copy.deepcopy(action)

        _, acc_sbc, sbc_distance_to_boundary = self.controller.step_func(
            dynamics=self.dynamics, acc_des=action, dt=self.dt, observation=sbc_data)

        if self.his_acc:
            if self.enable_sbc:
                obs_single_acc = np.concatenate([action, acc_sbc, self.dynamics.acc,
                                                 np.array([sbc_data['sbc_neighbor_aggressive']]),
                                                 np.array([sbc_data['sbc_obst_aggressive']])
                                                 ])
                self.obs_his_accs.append(obs_single_acc)
            else:
                obs_single_acc = np.concatenate([action, self.dynamics.acc])
                self.obs_his_accs.append(obs_single_acc)

        self.time_remain = self.ep_len - self.tick
        if self.enable_sbc:
            reward, rew_info = compute_reward_weighted(
                goal=self.goal, cur_pos=self.dynamics.pos, rl_acc=action,
                acc_sbc=acc_sbc, sbc_distance_to_boundary=sbc_distance_to_boundary,
                mellinger_acc=self.dynamics.acc, dt=self.control_dt,
                rew_coeff=self.rew_coeff, on_floor=self.dynamics.on_floor,
                action_prev=self.actions[1], aggressive_unclip=sbc_data['agg_unclip'], enable_sbc=self.enable_sbc
            )
        else:
            reward, rew_info = compute_reward_weighted(
                goal=self.goal, cur_pos=self.dynamics.pos, rl_acc=action,
                acc_sbc=None, sbc_distance_to_boundary=None,
                mellinger_acc=self.dynamics.acc, dt=self.control_dt,
                rew_coeff=self.rew_coeff, on_floor=self.dynamics.on_floor,
                action_prev=self.actions[1], aggressive_unclip=None, enable_sbc=self.enable_sbc
            )

        self.tick += 1
        done = self.tick > self.ep_len
        sv = self.state_vector(self)
        self.traj_count += int(done)

        acc_info = {'acc_ref': action, 'acc_sbc': acc_sbc, 'acc_real': self.dynamics.acc}
        return sv, reward, done, {'rewards': rew_info, 'acc': acc_info}

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
            self.dynamics_params = self.dyn_sampler_1.sample(
                self.dynamics_params)

        # Applying sampler 2
        if self.dyn_sampler_2 is not None:
            self.dynamics_params = self.dyn_sampler_2.sample(
                self.dynamics_params)

        # Checking that quad params make sense
        quad_rand.check_quad_param_limits(self.dynamics_params)

        # Updating params
        self.update_dynamics(dynamics_params=self.dynamics_params)

    def _reset(self, init_pos=None):
        # DYNAMICS RANDOMIZATION AND UPDATE
        # if self.dynamics_randomize_every is not None and (self.traj_count + 1) % self.dynamics_randomize_every == 0:
        #     self.resample_dynamics()

        if self.box < 10:
            self.box = self.box * self.box_scale
        x, y, z = self.np_random.uniform(-self.box,
                                         self.box, size=(3,)) + self.spawn_point

        if self.dim_mode == '1D':
            x, y = self.goal[0], self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]
        # Since being near the groud means crash we have to start above
        if z < 0.75:
            z = 0.75

        if init_pos is None:
            pos = npa(x, y, z)
        else:
            pos = np.array(init_pos)

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
                    box=(self.room_length, self.room_width, self.room_height),
                    vel_max=self.max_init_vel, omega_max=self.max_init_omega)
        else:
            # INIT HORIZONTALLY WITH 0 VEL and OMEGA
            vel, omega = np.zeros(
                3, dtype=np.float64), np.zeros(
                3, dtype=np.float64)

            if self.dim_mode == '1D' or self.dim_mode == '2D':
                rotation = np.eye(3)
            else:
                rotation = rotZ(0)[:3, :3]

        self.init_state = [pos, vel, rotation, omega]
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.dynamics.reset()
        self.dynamics.on_floor = False
        self.dynamics.crashed_floor = self.dynamics.crashed_wall = self.dynamics.crashed_ceiling = False

        # Reseting some internal state (counters, etc)
        self.tick = 0
        self.actions = [np.zeros([3, ]), np.zeros([3, ])]

        if self.his_acc:
            self.obs_his_accs = deque([], maxlen=self.his_acc_num)
            for _ in range(self.his_acc_num):
                self.obs_his_accs.append(np.zeros(self.obs_single_his_acc))

        state = self.state_vector(self)
        return state

    def reset(self, init_pos=None):
        return self._reset(init_pos)

    def render(self, **kwargs):
        """This class is only meant to be used as a component of QuadMultiEnv."""
        raise NotImplementedError()

    def step(self, action, sbc_data):
        return self._step(action, sbc_data)
