from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_single_baseline import QUAD_SINGLE_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ('quads_obs_repr', ['xyz_vxyz_R_omega_wall']),
])

QUAD_CLI_ADD_WALLS = QUAD_SINGLE_BASELINE_CLI + (
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=single_agent_baseline --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single_agent_add_walls',
    QUAD_CLI_ADD_WALLS,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_agent_add_walls', experiments=[_experiment])
