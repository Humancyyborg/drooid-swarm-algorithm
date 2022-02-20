from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_deepsets_obstacle_baseline import QUAD_8_OBSTACLES_PARAMETERZE_CLI

_params = ParamGrid([
    ('pbt_optimize_batch_size', [False]),
    ('quads_apply_downwash', [False]),
    ('quads_obstacle_size', [1.0]),
    ('quads_local_obst_obs', [2]),
    ('quads_obstacle_num', [6]),
])

BIG_MODEL_CLI = QUAD_8_OBSTACLES_PARAMETERZE_CLI + (
    ' --quads_obst_level_mode=0 --with_wandb=False --quads_obstacle_type=cylinder '
    '--hidden_size=256 --quads_neighbor_hidden_size=128 --quads_obstacle_hidden_size=128 '
    '--with_pbt=True --num_policies=8 --pbt_mix_policies_in_one_env=False --pbt_period_env_steps=10000000 '
    '--pbt_start_mutation=50000000 --pbt_mutation_rate=0.25 --pbt_replace_reward_gap=0.2 '
    '--pbt_replace_reward_gap_absolute=5.0 --pbt_optimize_gamma=True --pbt_perturb_max=1.2 '
    '--num_workers=72 --num_envs_per_worker=8 '
    '--quads_use_pos_diff=True'
)

_experiment = Experiment(
    '8pbt-no-downwash-big_model-256_128_128',
    BIG_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('6_obst_quads_multi_obst_mix_8a_v116', experiments=[_experiment])

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
