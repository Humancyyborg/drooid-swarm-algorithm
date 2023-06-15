from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_neighbor_visible_num", [2, 6]),
        ("quads_neighbor_encoder_type", ['attention', 'mean_embed', 'mlp']),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=36 --num_envs_per_worker=4 --quads_num_agents=8 '
    '--quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=test_neighbor_encoder'
)

_experiment = Experiment(
    "test_neighbor_encoder",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])
