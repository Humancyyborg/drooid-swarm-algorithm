# Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning

A codebase for training reinforcement learning policies for quadrotor swarms.
Includes:
* Flight dynamics simulator forked from https://github.com/amolchanov86/gym_art
and extended to support swarms of quadrotor drones
* Scripts and the necessary wrappers to facilitate training of control policies with Sample Factory
https://github.com/alex-petrenko/sample-factory

**Paper:** https://arxiv.org/abs/2109.07735

**Website:** https://sites.google.com/view/swarm-rl



<p align="middle">

<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Swarm_vs_Swarm.gif?raw=true" width="45%">
</p> 

<p align="middle">
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Obstacles_Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Scale_32_Static_Same_Goal.gif?raw=true" width="45%">
</p> 

## Installation

Initialize a Python environment, i.e. with `conda` (Python versions 3.6-3.8 are supported):

```
conda create -n swarm-rl python=3.8
conda activate swarm-rl
```

Then clone Sample Factory and install version 2.0:
```
git clone https://github.com/alex-petrenko/sample-factory.git
cd sample-factory
git checkout sf2
pip install -e .
```

Clone and install this repo as an editable Pip package:

```
git clone https://github.com/Zhehui-Huang/quad-swarm-rl.git
cd quad-swarm-rl
pip install -e .
```

This should pull and install all the necessary dependencies including PyTorch.

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
--max_policy_lag=100000000 --rnn_size=256 --gae_lambda=1.00 --max_grad_norm=5.0 \
--exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 --quads_use_numba=True \
--quads_mode=mix --quads_episode_duration=15.0 --quads_formation_size=0.0 \
--with_pbt=False --quads_collision_reward=5.0 \
--quads_neighbor_hidden_size=256 --neighbor_obs_type=pos_vel --quads_settle_reward=0.0 \
--quads_collision_hitbox_radius=2.0 --quads_collision_falloff_radius=4.0 --quads_local_obs=6 \
--quads_local_metric=dist --quads_local_coeff=1.0 --quads_num_agents=8 --quads_collision_reward=5.0 \
--quads_collision_smooth_max_penalty=10.0 --quads_neighbor_encoder_type=attention \
--replay_buffer_sample_prob=0.75 --anneal_collision_steps=300000000 --experiment=swarm_rl 
```

We also provide a training script `train.sh`, so you can simply start training by command `bash train.sh`.

Or, even better, you can use the runner scripts in `swarm_rl/runs/`. These runner scripts (a Sample Factory feature) are Python files that
contain experiment parameters, and support features such as evaluation on multiple seeds and gridsearches.

To execute a runner script run the following command:

```
python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline_attn --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
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
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME
```
EXPERIMENT_NAME and PATH_TO_TRAIN_DIR can be found in the cfg.json file of your trained model

## Unit Tests

To run unit tests:

```
./run_tests.sh
```

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our CORL paper.

```
@inproceedings{batra21corl,
  author    = {Sumeet Batra and
               Zhehui Huang and
               Aleksei Petrenko and
               Tushar Kumar and
               Artem Molchanov and
               Gaurav S. Sukhatme},
  title     = {Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning},
  booktitle = {5th Conference on Robot Learning, CoRL 2021, 8-11 November 2021, London, England, {UK}},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.07735}
}
```

Github issues and pull requests are welcome.
