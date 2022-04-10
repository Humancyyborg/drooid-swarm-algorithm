from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_deepsets_obstacle_baseline import QUADS_8_MULTI_NO_OBSTACLES_PARAMETERZE_CLI, seeds

_params = ParamGrid([
    ('seed', seeds(4)),
    ('quads_obs_repr', ['xyz_vxyz_R_omega']),
])

SMALL_MODEL_CLI = QUADS_8_MULTI_NO_OBSTACLES_PARAMETERZE_CLI + (
    ' --quads_num_agents=32 --quads_collision_reward=50.0 --quads_collision_smooth_max_penalty=100.0 '
    '--num_workers=18 --num_envs_per_worker=2 --reward_scale=0.05'
)

_experiment = Experiment(
    '32_agents-large_smooth_col',
    SMALL_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quad_multi_baseline', experiments=[_experiment])

# On Brain server, when you use num_workers = 72, if the system reports: Resource temporarily unavailable,
# then, try to use two commands below
# export OMP_NUM_THREADS=1
# export USE_SIMPLE_THREADED_LEVEL3=1

# Command to use this script on server:
# xvfb-run python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
# Command to use this script on local machine:
# Please change num_workers to the physical cores of your local machine
# python -m runner.run --run=quad_multi_mix_baseline --runner=processes --max_parallel=3 --pause_between=1 --experiments_per_gpu=1 --num_gpus=3
