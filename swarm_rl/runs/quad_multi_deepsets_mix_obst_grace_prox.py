from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_deepsets_obstacle_baseline import QUAD_OBSTACLE_PARAMETERZE_CLI

# quads_obstacle_num: 16
_params = ParamGrid([
    ('seed', [0000, 3333]),
    ('quads_obst_collision_enable_grace_period', [True]),
    ('quads_obst_level_change_cond', [2.0]),
    ('reward_scale', [0.1]),
    ('quads_collision_obst_smooth_max_penalty', [1.0, 2.0, 4.0, 6.0]),
])

_experiment = Experiment(
    'grace_inf_height_two_obst_prox_penalty_mix',
    QUAD_OBSTACLE_PARAMETERZE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('grace_quads_multi_obst_mix_8a_v116', experiments=[_experiment])

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
# srun --exclusive -c72 -N1 --gres=gpu:4 python -m sample_factory.runner.run --run=swarm_rl.runs.quad_multi_deepsets_through_random_obstacles --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
