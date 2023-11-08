from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_sbc_radius", [0.0, 0.05]),
        ("quads_obst_safe_coeff", [0.5, 1.0]),
        ("quads_neighbor_range", [1.5, 3.0]),
        ("quads_obst_range", [1.5, 3.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --quads_sbc_radius=0.0 --anneal_collision_steps=300000000 --quads_anneal_safe_start_steps=0 '
    '--quads_anneal_safe_total_steps=300000000 --quads_cost_rl_sbc=0.1 --quads_sbc_aggressive=0.2 '
    '--quads_cost_pos=1.0 --quads_cost_sbc_mellinger=0.0 '
    '--with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones --wandb_group=grid_search_sbc_coeff'
)

_experiment = Experiment(
    "grid_search_sbc_coeff_V1",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])