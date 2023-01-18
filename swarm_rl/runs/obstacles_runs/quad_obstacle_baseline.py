

QUAD_BASELINE_CLI_8 = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=10000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 '
    '--quads_neighbor_encoder_type=attention --replay_buffer_sample_prob=0.75 --save_milestones_sec=900 '
    '--normalize_input=False --normalize_returns=False --reward_clip=10 '
)