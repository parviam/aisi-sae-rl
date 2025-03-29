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
    --save_folder ~/p-adelarue3-0/impala_10env_25m_easy/ \
    --model_name v20250228_1011 \
    --total_timesteps_to_run 25000000 \
    --weight_update_per_save 200 \
    --iterations_per_weight_update 256 \
    --procgen_num_levels 10
```

### Continue Training
```bash
python train-ppo.py \
    --gym_env procgen:procgen-coinrun-v0 \
    --save_folder ~/p-adelarue3-0/impala_10env_25m_easy/ \
    --model_name v20250304_1748 \
    --total_timesteps_to_run 25000000 \
    --weight_update_per_save 200 \
    --iterations_per_weight_update 256 \
    --procgen_num_levels 10 \
    --continue_training True

```

## Experiment 2 (10M)

```bash
python train-ppo.py \
    --gym_env procgen:procgen-coinrun-v0 \
    --save_folder ~/p-adelarue3-0/impala_10level_v2/ \
    --model_name v20250306_1804 \
    --total_timesteps_to_run 25000000 \
    --weight_update_per_save 200 \
    --iterations_per_weight_update 256 \
    --procgen_num_levels 10 \
```

*Note `model_name` must be in camel-case**