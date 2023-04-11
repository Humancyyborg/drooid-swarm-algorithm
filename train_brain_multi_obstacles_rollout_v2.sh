python -m sample_factory.launcher.run \
--run=swarm_rl.runs.obstacles.quads_multi_obstacles_rollout_v2 \
--backend=slurm --slurm_workdir=slurm_output \
--experiment_suffix=slurm --pause_between=1 \
--slurm_gpus_per_job=1 --slurm_cpus_per_gpu=36 \
--slurm_sbatch_template=/home/zhehui/slurm/restart-swarm-rl_sbatch_timeout.sh \
--slurm_print_only=False