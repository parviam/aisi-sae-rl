import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys

# adding Folder_2 to the system path
sys.path.insert(0, '../../crosscoder-pipeline')
from baseline_utils import build_impala_cnn, learn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level,
             timesteps_per_proc, checkpoint_dir, save_interval,
             is_test_worker=False, comm=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    # Setup logger to store checkpoints in the designated checkpoint_dir.
    if checkpoint_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=checkpoint_dir, format_strs=format_strs)

    logger.info("creating environment")
    print(f"Environment Levels: {num_envs}\t{env_name}\t{num_levels}\t{distribution_mode}")

    venv = ProcgenEnv(num_envs=num_envs,
                      env_name=env_name,
                      num_levels=num_levels,
                      start_level=start_level,
                      distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16, 32, 32], emb_size=256)

    logger.info("training")
    learn(
        env=venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        model_name='ppo2_model',
        save_interval=save_interval,  # Save checkpoint every x training iterations.
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        # load_path="/data1/projects/keaton_gt_research/aisi-sae-rl/train-procgen/train_procgen/p-adelarue3-0/500_levels/checkpoints/00100",
    )

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000)
    # New argument for experiment name. The checkpoint directory is derived from this.
    parser.add_argument('--experiment_name', type=str, default='default_experiment')
    # New argument for save interval: how many training iterations between checkpoints.
    parser.add_argument('--save_interval', type=int, default=100)

    args = parser.parse_args()

    # Construct the checkpoint directory using the experiment name.
    checkpoint_dir = f"./p-adelarue3-0/{args.experiment_name}/"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

    train_fn(args.env_name,
             args.num_envs,
             args.distribution_mode,
             args.num_levels,
             args.start_level,
             args.timesteps_per_proc,
             checkpoint_dir,
             args.save_interval,
             is_test_worker=is_test_worker,
             comm=comm)

if __name__ == '__main__':
    main()
