from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 3333]),
    ('batch_size', [4096]),
    ('num_batches_per_epoch', [1, 2]),
])

SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --async_rl=False --serial_mode=True --num_workers=16 --num_envs_per_worker=2 --rollout=128 '
    '--with_wandb=True --wandb_project=HyperPPO --wandb_group=single_num_batch_size_v1 --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single_num_batch_size_v1',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])

# Command to use this script on local machine: Please change num_workers to the physical cores of your local machine
# python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad_batch_size --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
