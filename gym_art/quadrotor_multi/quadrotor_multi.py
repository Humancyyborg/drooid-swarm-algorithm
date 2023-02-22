import copy
import time
from collections import deque
from copy import deepcopy

import gym
import numpy as np

from gym_art.quadrotor_multi.quadrotor_multi_neighbors import MultiNeighbors
from gym_art.quadrotor_multi.quadrotor_multi_obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_room import Room
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario
from gym_art.quadrotor_multi.utils.quad_utils import all_dynamics, can_drones_fly

EPS = 1E-6


class QuadrotorEnvMulti(gym.Env):
    def __init__(self,
                 num_agents,
                 dynamics_params='DefaultQuad', dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr='xyz_vxyz_R_omega', ep_time=7, room_length=10, room_width=10,
                 room_height=10, init_random_state=False, rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False,
                 quads_mode='static_same_goal', quads_formation='circle_horizontal', quads_formation_size=-1.0,
                 swarm_obs='none', quads_use_numba=False, quads_view_mode='local',
                 collision_force=True, local_obs=-1, collision_hitbox_radius=2.0,
                 collision_falloff_radius=2.0, collision_smooth_max_penalty=10.0, use_replay_buffer=False,
                 vis_acc_arrows=False, viz_traces=25, viz_trace_nth_step=1,
                 use_obstacles=False, num_obstacles=0, obstacle_size=0.0, octree_resolution=0.05, use_downwash=False,
                 collision_obst_falloff_radius=3.0):

        super().__init__()
        # Prerequisite
        self.num_agents = num_agents
        self.swarm_obs = swarm_obs
        self.quads_mode = quads_mode
        self.apply_collision_force = collision_force

        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = (room_length, room_width, room_height)

        if local_obs == -1:
            self.num_use_neighbor_obs = num_agents - 1
        else:
            self.num_use_neighbor_obs = local_obs

        # Envs info
        self.envs = []
        self.quads_view_mode = quads_view_mode

        for i in range(num_agents):
            e = QuadrotorSingle(
                dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
                raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
                obs_repr, ep_time, room_length, room_width, room_height, init_random_state,
                rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification,
                quads_use_numba, swarm_obs, num_agents, quads_view_mode,
                self.num_use_neighbor_obs, use_obstacles=use_obstacles
            )
            self.envs.append(e)

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq

        # Reward info
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=5., quadcol_bin_smooth_max=10., quadcol_bin_obst=5., quadcol_bin_obst_smooth_max=10.,
            quadsettle=0., quadcol_coeff=1., quadcol_obst_coeff=1., crash_room=0.
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)

        if rew_coeff is not None:
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])

        orig_keys = list(rew_coeff_orig.keys())

        # Neighbor info
        self.neighbors = MultiNeighbors(obs_repr=obs_repr, obs_type=swarm_obs,
                                        visible_neighbor_num=self.num_use_neighbor_obs, use_downwash=use_downwash,
                                        collision_hitbox_radius=collision_hitbox_radius,
                                        collision_falloff_radius=collision_falloff_radius,
                                        collision_smooth_max_penalty=collision_smooth_max_penalty,
                                        num_agents=num_agents, control_freq=self.control_freq,
                                        rew_coeff=self.rew_coeff, observation_space=self.observation_space)

        # Obstacles info
        if use_obstacles and num_obstacles > 0:
            self.obstacles = MultiObstacles(num_obstacles=num_obstacles, room_dims=self.room_dims,
                                            resolution=octree_resolution, obstacle_size=obstacle_size,
                                            collision_obst_falloff_radius=collision_obst_falloff_radius,
                                            num_agents=num_agents, rew_coeff=self.rew_coeff, control_dt=self.control_dt)
        else:
            self.obstacles = None

        # Room info
        self.room = Room(num_agents=num_agents, rew_coeff=self.rew_coeff)

        # Scenarios info
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.set_room_dims,
                                        rew_coeff=self.rew_coeff,
                                        quads_formation=quads_formation, quads_formation_size=quads_formation_size)
        self.quads_formation_size = quads_formation_size

        # Render info
        self.simulation_start_time = 0
        self.frames_since_last_render = self.render_skip_frames = 0
        self.render_every_nth_frame = 1
        # set to below 1 slow motion, higher than 1 for fast-forward (if simulator can keep up)
        self.render_speed = 1.0
        self.all_collisions = {}

        # Scene info
        # set to true whenever we need to reset the OpenGL scene in render()
        # we don't actually create a scene object unless we want to render stuff
        self.scene = None
        self.reset_scene = False
        self.vis_acc_arrows = vis_acc_arrows
        self.viz_traces = viz_traces
        self.viz_trace_nth_step = viz_trace_nth_step

        # Replay info
        self.use_replay_buffer = use_replay_buffer
        # only start using the buffer after the drones learn how to fly
        self.activate_replay_buffer = False
        # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.saved_in_replay_buffer = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        # Valid check
        self.check_valid_input(local_obs, rew_coeff, orig_keys)

    def check_valid_input(self, local_obs, rew_coeff, orig_keys):
        assert local_obs <= self.num_agents - 1 or local_obs == -1, f'Invalid value ({local_obs}) passed to ' \
                                                                    f'--local_obs. Should be 0 < n < num_agents - 1, ' \
                                                                    f'or -1'
        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))

        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

    def set_room_dims(self, dims):
        # dims is a (x, y, z) tuple
        self.room_dims = dims

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        self.scene = Quadrotor3DSceneMulti(
            models=models,
            w=640, h=480, resizable=True, viewpoint=self.envs[0].viewpoint,
            room_dims=self.room_dims, num_agents=self.num_agents,
            render_speed=self.render_speed, formation_size=self.quads_formation_size, obstacles=self.obstacles,
            vis_acc_arrows=self.vis_acc_arrows, viz_traces=self.viz_traces, viz_trace_nth_step=self.viz_trace_nth_step,
        )

    def calculate_minor_info(self, rew_crash):
        # Render
        # Collisions with ground
        if self.scene is not None:
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            if self.obstacles is not None:
                obst_coll = [1.0 if i < 0 else 0.0 for i in self.obstacles.rew_obst_quad_collisions_raw]
            else:
                obst_coll = np.zeros(self.num_agents)

            self.all_collisions = {'drone': np.sum(self.neighbors.drone_col_matrix, axis=1),
                                   'ground': ground_collisions, 'obstacle': obst_coll}

        # Replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += rew_crash

    def reset(self, **kwargs):
        obs, rewards, dones, infos = [], [], [], []
        self.scenario.reset()
        self.quads_formation_size = self.scenario.formation_size

        # try to activate replay buffer if enabled
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            self.activate_replay_buffer = can_drones_fly(crashes_in_recent_episodes=self.crashes_in_recent_episodes)

        sense_positions = []
        sense_velocities = []
        goals = []
        for i, e in enumerate(self.envs):
            e.goal = self.scenario.goals[i]
            e.rew_coeff = self.rew_coeff
            e.update_env(*self.room_dims)

            observation = e.reset()

            # Get info
            obs.append(observation)
            sense_positions.append(e.sense_pos)
            sense_velocities.append(e.sense_vel)
            goals.append(e.goal)

        # extend obs to see neighbors
        obs = self.neighbors.reset(obs=obs, sense_positions=sense_positions, sense_velocities=sense_velocities,
                                   goals=goals, rew_coeff=self.rew_coeff)

        if self.obstacles is not None:
            obs = self.obstacles.reset(obs=obs, quads_pos=sense_positions, start_point=self.scenario.start_point,
                                       end_point=self.scenario.end_point, rew_coeff=self.rew_coeff)

        # Reset scene
        self.reset_scene = True
        # Reset variables for replay buffer
        self.crashes_last_episode = 0
        # Reset Collisions
        # All
        self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

        return obs

    # noinspection PyTypeChecker
    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        # Pre-set variables
        real_positions = np.array([env.dynamics.pos for env in self.envs])
        real_velocities = np.array([env.dynamics.vel for env in self.envs])
        real_rotations = np.array([env.dynamics.rot for env in self.envs])
        real_omegas = np.array([env.dynamics.omega for env in self.envs])

        sense_positions = [env.sense_pos for env in self.envs]
        sense_velocities = [env.sense_vel for env in self.envs]
        goals = [env.goal for env in self.envs]

        # 1) Calculate collision info
        # 1. neighbor; 2. obstacles; 3. room
        # 1. Neighbor
        self.neighbors.calculate_collision_info(rew_coeff=self.rew_coeff, real_positions=real_positions,
                                                cur_tick=self.envs[0].tick)
        # 2. Obstacles
        if self.obstacles is not None:
            self.obstacles.calculate_collision_info(rew_coeff=self.rew_coeff)

        # 3. Room
        apply_room_collision_flag = self.room.calculate_collision_info(envs=self.envs, rew_coeff=self.rew_coeff)

        # 2) Physical interaction
        # 1. neighbor; 2. obstacles; 3. room
        if self.apply_collision_force:
            # 1. Neighbor
            neighbor_velocities_change, neighbor_omegas_change = self.neighbors.perform_physical_interaction(
                real_positions=real_positions, real_velocities=real_velocities, real_rotations=real_rotations)

            # 2. Obstacles
            if self.obstacles is not None:
                obst_velocities_change, obst_omegas_change = self.obstacles.perform_physical_interaction(
                    real_positions=real_positions, real_velocities=real_velocities)
            else:
                obst_velocities_change = np.zeros(self.num_agents)
                obst_omegas_change = np.zeros(self.num_agents)

            # 3. Room
            if apply_room_collision_flag:
                new_velocities, new_omegas = self.room.perform_physical_interaction(
                    real_positions=real_positions, real_velocities=real_velocities, real_omegas=real_omegas,
                    room_box=self.envs[0].room_box)
            else:
                new_velocities = real_velocities
                new_omegas = real_omegas

            # Update vel & omega
            for i, env in enumerate(self.envs):
                env.dynamics.vel = new_velocities[i] + neighbor_velocities_change[i] + obst_velocities_change[i]
                env.dynamics.omega = new_omegas[i] + neighbor_omegas_change[i] + obst_omegas_change[i]

        # Calculate reward
        # 1. neighbor; 2. obstacles; 3. room
        # 1. Neighbor
        rew_collisions, rew_proximity = self.neighbors.calculate_reward()
        # 2. Obstacles
        if self.obstacles is not None:
            rew_collisions_obst_quad, rew_obst_quad_proximity = self.obstacles.calculate_reward(real_positions)
        else:
            rew_collisions_obst_quad = np.zeros(self.num_agents)
            rew_obst_quad_proximity = np.zeros(self.num_agents)

        # 3. Room
        rew_floor, rew_walls, rew_ceiling = self.room.calculate_reward()

        # Update goals
        self.scenario.step()

        # 3) Get final observations after collisions
        # 1. self; 2. neighbor; 2. obstacles;
        # 1. Self
        if apply_room_collision_flag:
            obs = [e.state_vector(e) for e in self.envs]

        # 2. Neighbor
        if self.num_use_neighbor_obs > 0:
            # extend obs to see neighbors
            # # Neighbor interaction
            obs = self.neighbors.step(obs=obs, sense_positions=sense_positions, sense_velocities=sense_velocities,
                                      goals=goals)

        # 3. Obstacles
        if self.obstacles is not None:
            obs = self.obstacles.step(obs=obs, sense_positions=sense_positions)

        # 4) Update rewards
        rewards += rew_collisions + rew_proximity + rew_collisions_obst_quad + rew_obst_quad_proximity + rew_floor + \
            rew_walls + rew_ceiling

        # 5) Log info
        # 1. neighbor; 2. obstacles; 3. room
        for i, info in enumerate(infos):
            # 1. Neighbor
            info["rewards"]["rew_quadcol"] = rew_collisions[i]
            info["rewards"]["rew_proximity"] = rew_proximity[i]

            # 3. Room
            info["rewards"]["rew_floor"] = rew_floor[i]
            info["rewards"]["rew_walls"] = rew_walls[i]
            info["rewards"]["rew_ceiling"] = rew_ceiling[i]

        if self.obstacles is not None:
            for i, info in enumerate(infos):
                info["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                info["rewards"]["rew_obst_quad_proximity"] = rew_obst_quad_proximity[i]

        # 6) Calculate minor info
        self.calculate_minor_info(rew_crash=infos[0]["rewards"]["rew_crash"])

        # DONES
        if any(dones):
            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.neighbors.collisions_per_episode,
                    }
                else:
                    infos[i]['episode_extra_stats'] = {
                        # Collision: add neighbor info
                        'num_collisions': self.neighbors.collisions_per_episode,
                        'num_collisions_after_settle': self.neighbors.collisions_after_settle,
                        f'num_collisions_{self.scenario.name()}': self.neighbors.collisions_after_settle,

                        # Collision: add room info
                        'num_collisions_with_room': self.room.collisions_room_per_episode,
                        'num_collisions_with_floor': self.room.collisions_floor_per_episode,
                        'num_collisions_with_walls': self.room.collisions_walls_per_episode,
                        'num_collisions_with_ceiling': self.room.collisions_ceiling_per_episode,

                    }
                    # Collision: add obstacles info
                    if self.obstacles is not None:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            self.obstacles.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats'][f'num_collisions_obst_{self.scenario.name()}'] = \
                            self.obstacles.obst_quad_collisions_per_episode

            obs = self.reset()
            dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

        return obs, rewards, dones, infos

    def render(self, mode='human', verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)

        if self.scene is None:
            self.init_scene_multi()

        if self.reset_scene:
            self.scene.update_models(models)
            self.scene.formation_size = self.quads_formation_size
            self.scene.update_env(self.room_dims)

            self.scene.reset(tuple(e.goal for e in self.envs), all_dynamics(envs=self.envs), self.obstacles,
                             self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            self.scene.formation_size = self.scenario.scenario.formation_size
        else:
            self.scene.formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        frame = self.scene.render_chase(all_dynamics=all_dynamics(envs=self.envs), goals=goals,
                                        collisions=self.all_collisions, mode=mode, obstacles=self.obstacles)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            self.scenario.scenario.update_formation_size(self.scene.formation_size)
        else:
            self.scenario.update_formation_size(self.scene.formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.envs[0].tick % 20 == 0:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

        if mode == "rgb_array":
            return frame

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory,
        # not here
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We need to reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
