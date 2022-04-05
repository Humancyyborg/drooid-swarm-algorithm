from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI, seeds

_params = ParamGrid([
    ('seed', seeds(4)),
])

SMALL_MODEL_CLI = QUAD_BASELINE_CLI + (
    ' --train_for_env_steps=10000000000 --hidden_size=16 --neighbor_obs_type=none --quads_local_obs=-1 '
    '--quads_num_agents=1 --replay_buffer_sample_prob=0.0 --anneal_collision_steps=0 --save_milestones_sec=10000 '
    '--quads_neighbor_encoder_type=no_encoder --with_wandb=True --wandb_user=multi-drone'
)

_experiment = Experiment(
    'baseline',
    SMALL_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quad_single_baseline', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
