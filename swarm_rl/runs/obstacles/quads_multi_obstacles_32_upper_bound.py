from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [2222, 3333]),
        ("quads_num_agents", [32]),
        ("quads_obst_density", [0.8]),
        ("quads_obst_size", [0.7]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=18 --num_envs_per_worker=2 --quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos_vel '
    '--quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=paper_search_obst_size_32'
)

_experiment = Experiment(
    "paper_search_obst_size_32",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
