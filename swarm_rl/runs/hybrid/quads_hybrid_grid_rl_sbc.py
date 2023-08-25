from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_cost_rl_sbc", [0.25, 0.5, 1.0, 2.0, 4.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --anneal_collision_steps=300000000 --quads_cost_pos=1.0 --quads_cost_sbc_mellinger=0.0 '
    '--with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones --wandb_group=grid_search_rl_sbc'
)

_experiment = Experiment(
    "grid_search_rl_sbc",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])