from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_deepsets_through_hole_obstacle import QUAD_OBSTACLE_BASELINE_CLI

_params = ParamGrid([
    ('seed', [2222]),
    ('quads_obstacle_num', [8]),
    ('quads_mode', ['through_hole']),
    ('quads_neighbor_encoder_type', ['mean_embed']),
    ('train_for_env_steps', [5000000000]),
    ('quads_local_obst_obs', [8]),
    ('obst_obs_type', ['pos_size', 'pos_vel_size_shape']),
    ('quads_obstacle_mode', ['static_door_fixsize', 'no_obstacles']),
])

_experiment = Experiment(
    'gridsearch-obst_obs_type-obst_mode',
    QUAD_OBSTACLE_BASELINE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_obst_through_hole_8a_v116', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_through_hole_obstacle --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_through_hole_obstacle --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4

# Slurm
# srun --exclusive -c72 -N1 --gres=gpu:4 python -m sample_factory.runner.run --run=swarm_rl.runs.quad_multi_deepsets_through_hole_obstacle_v5 --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
