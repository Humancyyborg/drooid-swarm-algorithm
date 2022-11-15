import copy

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.additional_input import QuadsAdditionalInputWrapper
from swarm_rl.env_wrappers.discrete_actions import QuadsDiscreteActionsWrapper
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper, \
    DEFAULT_QUAD_REWARD_SHAPING_SINGLE
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = cfg.quads_episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    extended_obs = cfg.neighbor_obs_type

    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents,
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True, ep_time=episode_duration, room_length=cfg.room_dims[0],
        room_width=cfg.room_dims[1], room_height=cfg.room_dims[2], rew_coeff=rew_coeff,
        quads_mode=cfg.quads_mode, quads_formation=cfg.quads_formation, quads_formation_size=cfg.quads_formation_size,
        swarm_obs=extended_obs, quads_use_numba=cfg.quads_use_numba,
        quads_view_mode=cfg.quads_view_mode, local_obs=cfg.quads_local_obs, obs_repr=cfg.quads_obs_repr,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius, collision_falloff_radius=cfg.quads_collision_falloff_radius,
        use_replay_buffer=use_replay_buffer,
    )

    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty, cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing)
    env = QuadEnvCompatibility(env, render_mode=render_mode)
    return env


def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    if env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    else:
        raise NotImplementedError
