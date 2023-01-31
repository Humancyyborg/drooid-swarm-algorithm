from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.quad_multi_mix_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    '--train_for_env_steps=250000000 --num_workers=4 --use_obstacles=False --use_downwash=True '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=downwash --wandb_user=multi-drones '
    '--anneal_collision_steps=300000000'
)

_experiment = Experiment(
    "baseline_multi_drone",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("quad_multi_downwash", experiments=[_experiment])
# python -m sample_factory.launcher.run --run=swarm_rl.runs.sf2_multi_drone --runner=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=4 --num_gpus=1
