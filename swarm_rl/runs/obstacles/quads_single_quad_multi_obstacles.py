from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_num_agents", [1]),
        ("quads_obst_size", [0.6, 0.7]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=72 --num_envs_per_worker=16 --quads_obst_density=0.8 --save_milestones_sec=1200 '
    '--quads_neighbor_visible_num=0 --quads_neighbor_obs_type=none --quads_encoder_type=corl '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=single_drone_multi_obst'
)

_experiment = Experiment(
    "single_drone_multi_obst",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("single_drone_multi_obst", experiments=[_experiment])