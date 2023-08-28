from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

from swarm_rl.utils import timeStamped

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
])

_experiment = Experiment(
    'quad_mix_baseline-8_mixed_attn',
    QUAD_BASELINE_CLI_8,
    _params.generate_params(randomize=False),
)

run_name = timeStamped("test_anneal", fmt="{fname}_%Y%m%d_%H%M")

RUN_DESCRIPTION = RunDescription(run_name, experiments=[_experiment])

# For scale, need to change
# num_workers / num_envs_per_worker && quads_num_agents
# num_workers * num_envs_per_worker * quads_num_agents should not change