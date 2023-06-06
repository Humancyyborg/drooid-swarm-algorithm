from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [8]),
        ("quads_obstacle_obs_type", ["octomap"]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 --quads_obstacle_visible_num=2 '
    '--quads_encoder_type=corl --quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos_vel '
    '--quads_obst_encoder_type=mlp --replay_buffer_sample_prob=0.0 '
    '--quads_neighbor_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=paper_old_obst_obs'
)

_experiment = Experiment(
    "paper_old_obst_obs",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])