from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid([
    ('seed', [0000, 3333]),
])

MULTI_8_CLI = QUAD_BASELINE_CLI_8 + (
    ' --quads_neighbor_encoder_type=attention --with_wandb=True --wandb_project=Quad-Swarm-RL '
    '--wandb_group=restart_refactor_v2 --wandb_user=multi-drones'
)

_experiment = Experiment(
    'restart_refactor_v2',
    MULTI_8_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])

# Command to use this script on local machine: Please change num_workers to the physical cores of your local machine
# python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
