from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_cost_sbc_mellinger", [0.5, 1.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones '
    '--wandb_group=grid_search_reward'
)

_experiment = Experiment(
    "grid_search_reward",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])