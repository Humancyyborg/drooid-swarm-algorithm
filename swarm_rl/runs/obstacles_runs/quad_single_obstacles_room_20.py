from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles_runs.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111]),
        ("quads_obst_collision_reward", [0.0, 2.5, 5.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    '--train_for_env_steps=1000000000 --num_workers=36 --num_envs_per_worker=8 '
    '--use_obstacles=True --quads_obstacle_density=0.25 --quads_obstacle_size=0.6 '
    '--quads_collision_obst_falloff_radius=3.0 '
    '--quads_obst_collision_smooth_max_penalty=0.0 '
    '--quads_neighbor_hidden_size=256 --neighbor_obs_type=none --quads_neighbor_encoder_type=no_encoder '
    '--quads_local_obs=-1 --quads_num_agents=1 '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=one_drone_multi_obst_new_octomap_room_20 '
    '--wandb_user=multi-drones --anneal_collision_steps=300000000 --replay_buffer_sample_prob=0 --room_dims 20 20 20'
)

_experiment = Experiment(
    "one_drone_multi_obst_new_octomap_room_20",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("test_new_obst_octomap_room_20", experiments=[_experiment])
