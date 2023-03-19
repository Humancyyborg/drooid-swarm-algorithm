from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('quads_neighbor_encoder_type', ['attention']),
    ('seed', [0000]),
    ('num_workers', [16]),
    ('num_envs_per_worker', [4]),
])

_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])

# Command to use this script on local machine: Please change num_workers to the physical cores of your local machine
# python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
