import copy

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


class TwoStageAnnealSchedule:
    def __init__(self, coeff_name, final_value, start_steps, total_steps, start_value=0.0):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.start_steps = start_steps
        self.total_steps = total_steps
        self.start_value = start_value


def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None
    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr, his_acc=cfg.quads_obs_acc_his, his_acc_num=cfg.quads_obs_acc_his_num,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area,

        # Aerodynamics
        use_downwash=cfg.quads_use_downwash,
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=cfg.quads_mode,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False,
        # Rendering
        render_mode=render_mode,
        # SBC specific
        sbc_radius=cfg.quads_sbc_radius,
        sbc_nei_range=cfg.quads_neighbor_range, sbc_obst_range=cfg.quads_obst_range,
        sbc_max_acc=cfg.quads_max_acc,
        sbc_max_neighbor_aggressive=cfg.quads_max_neighbor_aggressive,
        sbc_max_obst_aggressive=cfg.quads_max_obst_aggressive,
        sbc_max_room_aggressive=cfg.quads_max_room_aggressive,
    )

    if use_replay_buffer:
        env = ExperienceReplayWrapper(
            env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
            cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
            cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    reward_shaping['quad_rewards']['pos'] = cfg.quads_cost_pos
    reward_shaping['quad_rewards']['crash'] = cfg.quads_cost_crash
    reward_shaping['quad_rewards']['act_change'] = cfg.quads_cost_act_change
    reward_shaping['quad_rewards']['cbg_agg'] = cfg.quads_cost_cbf_agg

    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward

    reward_shaping['quad_rewards']['rl_sbc'] = cfg.quads_cost_rl_sbc
    reward_shaping['quad_rewards']['rl_mellinger'] = cfg.quads_cost_rl_mellinger
    reward_shaping['quad_rewards']['sbc_boundary'] = cfg.quads_sbc_boundary

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0

        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    # this is annealed by the reward shaping wrapper
    if cfg.quads_anneal_safe_total_steps > 0:
        # rl_sbc & rl_mellinger
        reward_shaping['quad_rewards']['rl_sbc'] = 0.0
        reward_shaping['quad_rewards']['rl_mellinger'] = 0.0
        reward_shaping['quad_rewards']['sbc_boundary'] = 0.0

        safe_annealing = [
            TwoStageAnnealSchedule(
                coeff_name='rl_sbc', final_value=cfg.quads_cost_rl_sbc, start_steps=cfg.quads_anneal_safe_start_steps,
                total_steps=cfg.quads_anneal_safe_total_steps, start_value=0.0),
            TwoStageAnnealSchedule(
                coeff_name='rl_mellinger', final_value=cfg.quads_cost_rl_mellinger,
                start_steps=cfg.quads_anneal_safe_start_steps, total_steps=cfg.quads_anneal_safe_total_steps,
                start_value=0.0),
            TwoStageAnnealSchedule(
                coeff_name='sbc_boundary', final_value=cfg.quads_sbc_boundary,
                start_steps=cfg.quads_anneal_safe_start_steps, total_steps=cfg.quads_anneal_safe_total_steps,
                start_value=0.0),
        ]
    else:
        safe_annealing = None

    if cfg.cbf_agg_anneal_steps > 0:
        reward_shaping['quad_rewards']['sbc_nei_max_agg'] = 0.01
        reward_shaping['quad_rewards']['sbc_obst_max_agg'] = 0.01

        cbf_aggressive_annealing = [
            TwoStageAnnealSchedule(
                coeff_name='sbc_nei_max_agg', final_value=1.0,
                start_steps=0, total_steps=cfg.cbf_agg_anneal_steps,
                start_value=0.01),
            TwoStageAnnealSchedule(
                coeff_name='sbc_obst_max_agg', final_value=1.0,
                start_steps=0, total_steps=cfg.cbf_agg_anneal_steps,
                start_value=0.01),
        ]
    else:
        cbf_aggressive_annealing = None
        reward_shaping['quad_rewards']['sbc_nei_max_agg'] = 1.0
        reward_shaping['quad_rewards']['sbc_obst_max_agg'] = 1.0

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    safe_annealing=safe_annealing, cbf_aggressive_annealing=cbf_aggressive_annealing,
                                    with_pbt=cfg.with_pbt)
    env = QuadEnvCompatibility(env)

    if cfg.visualize_v_value:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        env = V_ValueMapWrapper(env, actor_critic)

    return env


def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    if env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    else:
        raise NotImplementedError
