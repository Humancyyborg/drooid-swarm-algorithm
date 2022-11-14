#!/bin/bash
source /home/zhehui/miniconda3/etc/profile.d/conda.sh
conda activate swarm-rl

timeout $TIMEOUT $CMD
if [[ $$? -eq 124 ]]; then
    sbatch $PARTITION--gres=gpu:$GPU -c $CPU --parsable --output $FILENAME-slurm-%j.out $FILENAME
fi
