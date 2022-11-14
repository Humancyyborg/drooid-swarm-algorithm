#!/bin/bash
conda activate swarm-rl

timeout $TIMEOUT $CMD
if [[ $$? -eq 124 ]]; then
    sbatch $PARTITION--gres=gpu:$GPU -c $CPU --parsable --output $FILENAME-slurm-%j.out $FILENAME
fi
