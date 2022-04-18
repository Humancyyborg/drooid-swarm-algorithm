from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_deepsets_obstacle_baseline import QUAD_8_OBSTACLES_PARAMETERZE_CLI, seeds

_params = ParamGrid([
    ('seed', seeds(4)),
    ('quads_early_termination', [True]),
    ('quads_init_random_state', [False]),
])

SMALL_MODEL_CLI = QUAD_8_OBSTACLES_PARAMETERZE_CLI + (
    ' --quads_local_obst_obs=2 --quads_obst_level_mode=1 --quads_obstacle_num=10 --quads_episode_duration=20.0 '
    '--quads_neighbor_proximity_mode=1 --quads_obst_proximity_mode=1 '
    '--hidden_size=256 --quads_neighbor_hidden_size=128 --quads_obstacle_hidden_size=128'
)

_experiment = Experiment(
    'spawn_in_air_early_term',
    SMALL_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('8_multi_2_local', experiments=[_experiment])