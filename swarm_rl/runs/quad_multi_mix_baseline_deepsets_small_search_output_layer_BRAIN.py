from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8_BRAIN

_params = ParamGrid([
    ('quads_neighbor_encoder_type', ['mean_embed']),
    ('seed', [1111, 2222]),
    ('train_for_env_steps', [5000000000]),
    ('anneal_collision_steps', [0]),
    ('hidden_size', [8]),
    ('quads_neighbor_hidden_size', [4]),
    ('quads_output_hidden_size', [8, 16]),
])

_experiment = Experiment(
    'brain_grid_search_output_hid_size',
    QUAD_BASELINE_CLI_8_BRAIN,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('brain_output_hid_size_8a_deepsets_small_v116', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
