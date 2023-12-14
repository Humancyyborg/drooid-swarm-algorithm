from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.hybrid.baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("anneal_collision_steps", [0, 300000000]),
        ("quads_anneal_safe_total_steps", [0, 300000000]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # Self
    ' --quads_num_agents=8 --quads_obs_repr=xyz_vxyz_R_omega_floor_acc --quads_episode_duration=20.0 '
    '--quads_obs_acc_his=False --quads_obs_acc_his_num=0 --quads_max_acc=2.0 '
    # Obstacle
    '--quads_obst_density=0.8 --quads_obst_size=0.85 '
    # SBC
    '--quads_cost_rl_sbc=0.1 --quads_cost_rl_mellinger=0.1 --quads_sbc_boundary=0.0 --quads_sbc_radius=0.05 '
    '--quads_max_neighbor_aggressive=5.0 --quads_max_obst_aggressive=5.0 --quads_max_room_aggressive=0.2 '
    '--quads_neighbor_range=2.0 --quads_obst_range=2.0 '
    # Annealing
    # '--anneal_collision_steps=0 '
    # Safe Annealing
    '--quads_anneal_safe_start_steps=0 '
    # '--quads_anneal_safe_start_steps=0 --quads_anneal_safe_total_steps=0 '
    # Wandb
    '--with_wandb=True --wandb_project=Quad-Hybrid --wandb_user=multi-drones '
    '--wandb_group=grid_search_anneal'
)

_experiment = Experiment(
    "grid_search_anneal",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("hybrid", experiments=[_experiment])