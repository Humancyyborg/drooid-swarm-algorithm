from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

from swarm_rl.runs.quad_multi_deepsets_obstacle_baseline import QUAD_8_OBSTACLES_PARAMETERZE_CLI, seeds

_params = ParamGrid([
    ('seed', seeds(2)),
    ('quads_obs_repr', ['xyz_vxyz_R_omega']),
    ('quads_collision_smooth_max_penalty', [0.1, 4.0]),
])

BIG_MODEL_CLI = QUAD_8_OBSTACLES_PARAMETERZE_CLI + (
    ' --hidden_size=256 --quads_neighbor_hidden_size=128 --quads_obstacle_hidden_size=128'
)

_experiment = Experiment(
    'obs_repr-col_smooth_pen-big_model',
    BIG_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('8_obst_2_local_quads_multi_obst_mix_8a_v116', experiments=[_experiment])