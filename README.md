# Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning

A codebase for training reinforcement learning policies for quadrotor swarms.
Includes:
* Flight dynamics simulator forked from https://github.com/amolchanov86/gym_art
and extended to support swarms of quadrotor drones
* Scripts and the necessary wrappers to facilitate training of control policies with Sample Factory
https://github.com/alex-petrenko/sample-factory

**Paper:** https://openreview.net/pdf?id=ofioIEZvJRG

**Website:** https://sites.google.com/view/swarm-rl

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&emsp;&emsp;&emsp; Same Goal (8 drones, 1.5x real time) &nbsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Swarm vs Swarm (8 drones, 1.5x real time)

<p align="middle">
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Static_Same_Goal.gif?raw=true" width="400">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Swarm_vs_Swarm.gif?raw=true" width="400">
</p> 

One Dynamic Obstacle + Same Goal (8 drones, 1.5x real time) &nbsp;&nbsp;&emsp;&emsp; Same Goal (32 drones, 1.5x real time)

<p align="middle">
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Obstacles_Static_Same_Goal.gif?raw=true" width="400">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Scale_32_Static_Same_Goal.gif?raw=true" width="400">
</p> 

## Installation

Initialize a Python environment, i.e. with `conda` (Python versions 3.6-3.8 are supported):

```
conda create -n swarm-rl python=3.8
conda activate swarm-rl
```

Clone and install this repo as an editable Pip package:

```
git clone https://github.com/alex-petrenko/quad-swarm-rl
cd quad-swarm-rl
pip install -e .
```

This should pull and install all the necessary dependencies, including Sample Factory and PyTorch.

## Running experiments

### Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

```
python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO \
--use_rnn=False \
--num_workers=36 --num_envs_per_worker=4 \
--learning_rate=0.0001 --ppo_clip_value=5.0 \
--recurrence=1 --nonlinearity=tanh --actor_critic_share_weights=False \
--policy_initialization=xavier_uniform --adaptive_stddev=False --with_vtrace=False \
--max_policy_lag=100000000 --hidden_size=256 --gae_lambda=1.00 --max_grad_norm=5.0 \
--exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 --quads_use_numba=True \
--quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 \
--encoder_custom=quad_multi_encoder --with_pbt=False --quads_collision_reward=5.0 \
--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel --quads_settle_reward=0.0 \
--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 --quads_local_obs=6 \
--quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 --quads_collision_reward=5.0 \
--quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=attention \
--replay_buffer_sample_prob=0.75 --anneal_collision_steps=300000000 --experiment=swarm_rl 
```

Or, even better, you can use the runner scripts in `swarm_rl/runs/`. Runner scripts (a Sample Factory feature) are Python files that
contain experiment parameters, and support features such as evaluation on multiple seeds and gridsearches.

To execute a runner script run the following command:

```
python -m sample_factory.runner.run --run=swarm_rl.runs.quad_multi_mix_baseline_attn --runner=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

This command will start training four different seeds in parallel on a 4-GPU server. Adjust the parameters accordingly to match
your hardware setup.

To monitor the experiments, go to the experiment folder, and run the following command:

```
tensorboard --logdir=./
```

### Test
To test the trained model, run the following command:

```
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --continuous_actions_sample=False --quads_use_numba=False --train_dir=PATH_TO_PROJECT/swarm_rl/train_dir --experiments_root=EXPERIMENT_ROOT --experiment=EXPERIMENT_NAME
```

## Unit Tests

To run unit tests:

```
./run_tests.sh
```