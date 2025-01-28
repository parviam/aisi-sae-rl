# aisi-sae-rl


## Install
To setup the conda environment with the required python packages, use environment.yml like:

    conda env create -n rl_env --file=environment.yml

## Activate
conda activate rl_env

## Run Pipeline with ProcGen Environment

```bash
cd training-pipeline

python train-ppo.py \
    --gym_env procgen:procgen-coinrun-v0 \
    --save_folder ~/scratch/aisa-testing/testing \
    --model_name dummy \
    --total_timesteps_to_run 50000 \
    --weight_update_per_save 5 \
    --iterations_per_weight_update 10
```