from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444]),
    ('quads_obstacle_mode', ['static_door_fixsize']),
    ('quads_obstacle_num', [8]),
    ('quads_mode', ['through_hole']),
    ('quads_neighbor_encoder_type', ['mean_embed']),
])

QUAD_OBSTACLE_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=through_hole --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=256 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=8 --quads_local_obst_obs=8 --quads_obstacle_mode=static_door_fixsize '
    '--obst_obs_type=pos_vel_size'
)

QUAD_OBSTACLE_THROUGH_HOLE_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=through_hole --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=256 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=8 --quads_local_obst_obs=8 --quads_obstacle_mode=static_door_fixsize '
    '--obst_obs_type=pos_vel_size --room_length=3.0 --room_height=3.0 --room_width=10.0'
)

QUAD_OBSTACLE_RANDOM_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=through_random_obstacles --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=256 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=8 --quads_local_obst_obs=8 --quads_obstacle_mode=static_random_place_fixsize '
    '--obst_obs_type=pos_vel_size --room_length=4.0 --room_height=4.0 --room_width=10.0'
)

QUAD_OBSTACLE_PILLAR_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=o_dynamic_same_goal --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=256 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=4 --quads_local_obst_obs=4 --quads_obstacle_mode=static_pillar_fixsize '
    '--obst_obs_type=pos_size --room_length=10.0 --room_height=10.0 --room_width=10.0 --save_milestones_sec=1000'
)

QUAD_OBSTACLE_PILLAR_TWO_STACKS_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=5000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=o_dynamic_same_goal --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=256 --quads_obst_penalty_fall_off=10.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=8 --quads_local_obst_obs=8 --quads_obstacle_mode=static_pillar_fixsize '
    '--obst_obs_type=pos_size --room_length=10.0 --room_width=10.0 --room_height=4.0 --save_milestones_sec=1000'
)

QUAD_OBSTACLE_PILLAR_TWO_STACKS_SIX_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=2000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=o_dynamic_same_goal --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=128 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=128 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=6 --quads_local_obst_obs=6 --quads_obstacle_mode=static_pillar_fixsize '
    '--obst_obs_type=pos_size --room_length=10.0 --room_width=10.0 --room_height=3.0 --quads_obstacle_stack_num=3 '
    '--save_milestones_sec=1000 --reward_scale=0.25 --quads_obst_level_mode=1 --quads_enable_sim_room=none '
    '--quads_obst_proximity_mode=1'
)

QUAD_OBSTACLE_PARAMETERZE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=2500000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=128 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=10.0 '
    '--quads_obstacle_hidden_size=128 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=2 --quads_local_obst_obs=-1 --quads_obstacle_mode=static_pillar_fixsize '
    '--obst_obs_type=pos_size --room_length=10.0 --room_width=10.0 --room_height=10.0 --quads_obstacle_stack_num=2 '
    '--save_milestones_sec=10000 --reward_scale=0.25 --quads_obst_level_mode=0 --quads_enable_sim_room=wall-ceiling '
    '--quads_obst_proximity_mode=0 --quads_obst_inf_height=True'
)

QUAD_OBSTACLE_PARAMETERZE_LONG_DURATION_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=2000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=40.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=128 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=2.0 '
    '--quads_obstacle_hidden_size=128 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=2 --quads_local_obst_obs=-1 --quads_obstacle_mode=static_pillar_fixsize '
    '--obst_obs_type=pos_size --room_length=10.0 --room_width=10.0 --room_height=10.0 --quads_obstacle_stack_num=2 '
    '--save_milestones_sec=10000 --reward_scale=0.05 --quads_obst_level_mode=0 --quads_enable_sim_room=wall-ceiling '
    '--quads_obst_proximity_mode=0 --quads_obst_inf_height=True --quads_obst_collision_enable_grace_period=True '
    '--quads_obst_level_change_cond=6.0 --gamma=0.998'
)

# 4 OBSTACLES
QUAD_OBSTACLE_PBT_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=10000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --hidden_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_mode=mix --quads_episode_duration=40.0 --quads_formation_size=0.0 '
    '--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 '
    '--quads_neighbor_hidden_size=128 --neighbor_obs_type=pos_vel '
    '--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 '
    '--quads_local_obs=6 --quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 '
    '--quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=mean_embed '
    '--replay_buffer_sample_prob=0.0 --anneal_collision_steps=0.0 --quads_obstacle_type=cube '
    '--quads_collision_obstacle_reward=5.0 --quads_collision_obst_smooth_max_penalty=2.0 '
    '--quads_obstacle_hidden_size=128 --quads_obst_penalty_fall_off=3.0 --quads_obstacle_size=1.0 '
    '--quads_obstacle_num=4 --quads_local_obst_obs=-1 --quads_obstacle_mode=static_pillar_fixsize '
    '--obst_obs_type=pos_size --room_length=10.0 --room_width=10.0 --room_height=10.0 --quads_obstacle_stack_num=2 '
    '--save_milestones_sec=10000 --reward_scale=0.1 --quads_obst_level_mode=0 --quads_enable_sim_room=wall-ceiling '
    '--quads_obst_proximity_mode=0 --quads_obst_inf_height=True --quads_obst_collision_enable_grace_period=True '
    '--quads_obst_level_change_cond=6.0'
)


_experiment = Experiment(
    'neighbor_deepsets_cube_fix_size',
    QUAD_OBSTACLE_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_obst_through_hole_8a_v116', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_through_hole_obstacle --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_through_hole_obstacle --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
