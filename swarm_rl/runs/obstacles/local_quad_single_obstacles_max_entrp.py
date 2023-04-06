from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000]),
        ('exploration_loss_coeff', [0.003]),
        ('max_entropy_coeff', [0.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_workers=12 --num_envs_per_worker=4 --quads_num_agents=1 '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=obst_quad_single_max_entrop_0'
)

_experiment = Experiment(
    "obst_quad_single_max_entrop_0",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_single", experiments=[_experiment])
