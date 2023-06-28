from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 3333]),
])

SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --async_rl=False --serial_mode=True --num_workers=16 --num_envs_per_worker=2 --rollout=128 --batch_size=2048 '
    '--num_batches_per_epoch=4 '
    '--with_wandb=True --wandb_project=HyperPPO --wandb_group=single --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('paper_quads_multi_mix_baseline_8a_attn_v116', experiments=[_experiment])
