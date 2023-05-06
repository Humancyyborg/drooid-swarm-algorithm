from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 3333]),
    ('quads_sim_steps', [4, 8]),
])

SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --quads_sim_freq=200 --with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_group=single_sim_freq --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single_sim_freq_4_8',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_sim_freq', experiments=[_experiment])
