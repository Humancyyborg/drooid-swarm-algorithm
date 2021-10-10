from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444]),
    ('quads_obstacle_mode', ['static_door_fixsize']),
    ('quads_obstacle_num', [8]),
    ('quads_obstacle_type', ['cube']),
    ('quads_collision_obstacle_reward', [5.0]),
    ('quads_obstacle_obs_mode', ['relative']),
    ('quads_collision_obst_smooth_max_penalty', [10.0]),
    ('quads_obstacle_hidden_size', [256]),
    ('replay_buffer_sample_prob', [0.0]),
    ('quads_obst_penalty_fall_off', [10.0]),
    ('quads_obstacle_size', [1.0]),
    ('quads_mode', ['through_hole']),
    ('anneal_collision_steps', [0.0]),
])

_experiment = Experiment(
    'cube_fix_size',
    QUAD_BASELINE_CLI,
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
