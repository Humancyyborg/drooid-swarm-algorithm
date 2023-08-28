from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("with_pbt", ["True"]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # PBT
    ' --num_policies=8 --pbt_mix_policies_in_one_env=True --pbt_period_env_steps=10000000 '
    '--pbt_start_mutation=50000000 --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 '
    '--pbt_optimize_gamma=True --pbt_perturb_max=1.2 '
    # Pre-set hyperparameters
    '--exploration_loss_coeff=0.0005 --max_entropy_coeff=0.0005 '
    '--anneal_collision_steps=0 --train_for_env_steps=10000000000 '
    # Num workers
    '--num_workers=68 --num_envs_per_worker=2 --quads_num_agents=8 '
    # Neighbor & General Encoder for obst & neighbor
    '--quads_neighbor_visible_num=6 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    # WandB
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=pbt_obstacle_multi_attn_v2'
)

_experiment = Experiment(
    "pbt_obstacle_multi_attn_v2",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])