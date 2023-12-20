 We demonstrate decentralized drone swarm control learned via large-scale multi-agent reinforcement learning. Neural network policies trained to control individual drones in physics simulation produce advanced flocking, tight maneuvers in formation, collision avoidance, dynamic formation restructuring to dodge obstacles, and pursuit task coordination. By analyzing model architectures & training parameters influencing final performance, we develop policies that transfer from simulation to physically constrained drones, enabling key swarm behaviors like station keeping and goal swapping within constrained environments.

**Website:**https://drooid.xyz/



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

We provide a training script `train.sh`, so you can simply start training by command `bash train.sh`.

Or, even better, you can use the runner scripts in `swarm_rl/runs/`. These runner scripts (a Sample Factory feature) are Python files that
contain experiment parameters, and support features such as evaluation on multiple seeds and gridsearches.

To execute a runner script run the following command:

```
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

This command will start training four different seeds in parallel on a 4-GPU server. Adjust the parameters accordingly to match
your hardware setup.

To monitor the experiments, go to the experiment folder, and run the following command:

```
tensorboard --logdir=./
```
### WandB support

If you want to monitor training with WandB, follow the steps below: 
- setup WandB locally by running `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb).
* add `--with_wandb=True` in the command.

Here is a total list of wandb settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

### Test
To test the trained model, run the following command:

```
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode CAMERA_VIEWS
```
EXPERIMENT_NAME and PATH_TO_TRAIN_DIR can be found in the cfg.json file of your trained model

CAMERA_VIEWS can be any number of views from the following: `[topdown, global, chase, side, corner0, corner1, corner2, corner3, topdownfollow]`


## Unit Tests

To run unit tests:

```
./run_tests.sh
```

## Reference 

https://arxiv.org/abs/2109.07735

