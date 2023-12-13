from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_max_acc", [6.0]),
        ("quads_max_neighbor_aggressive", [50.0, 100.0, 200.0]),
        ("quads_max_obst_aggressive", [50.0, 100.0, 200.0]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --anneal_collision_steps=300000000 --quads_anneal_safe_start_steps=0 --quads_anneal_safe_total_steps=300000000 '
    '--quads_num_agents=8 --num_workers=36 --num_envs_per_worker=4 --quads_obst_density=0.8 --quads_obst_size=0.85 '
    '--quads_cost_rl_sbc=0.1 --quads_cost_pos=1.0 --quads_cost_sbc_mellinger=0.0 --quads_neighbor_range=3.0 '
    '--quads_obst_range=3.0 --quads_sbc_radius=0.05 --quads_sbc_boundary=0.1 '
    '--with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones '
    '--wandb_group=quad_8_grid_search_sbc_aggressive'
)

_experiment = Experiment(
    "quad_8_grid_search_sbc_aggressive",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])