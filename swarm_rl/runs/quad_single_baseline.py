from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

QUAD_SINGLE_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=16 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=16.0 --quads_formation_size=0.0 '
    '--with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=none --quads_neighbor_encoder_type=no_encoder '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=-1 --quads_num_agents=1 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 '
    '--replay_buffer_sample_prob=0.75 '
    '--anneal_collision_steps=300000000 '
)


CLI = QUAD_SINGLE_BASELINE_CLI + (
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=single_agent_baseline --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single_agent_baseline',
    CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_agent_baseline', experiments=[_experiment])
