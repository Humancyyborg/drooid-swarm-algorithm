from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_deepsets_obstacle_baseline import QUAD_OBSTACLE_PBT_CLI

_params = ParamGrid([
    ('pbt_optimize_batch_size', [False]),
])

PBT_CLI = QUAD_OBSTACLE_PBT_CLI + (
    ' --with_pbt=True --num_policies=8 --pbt_mix_policies_in_one_env=True --pbt_period_env_steps=10000000 '
    '--pbt_start_mutation=50000000 --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=100.0 '
    '--num_workers=72 --num_envs_per_worker=16'
)

_experiment = Experiment(
    'quad_mix_obst_baseline-8_pbt',
    PBT_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_obst_baseline_pbt8_v116', experiments=[_experiment])
