import argparse
import os

import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from tensorboardX import SummaryWriter

from impala import ImpalaActorCriticPolicy
from wrapped_procgen import make_wrapped_procgen_env
from utils import get_filename_with_highest_timestep


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PPO Training Script")

    # GYM ENV
    parser.add_argument("--gym_env", type=str, default="CartPole-v1", 
                        help="Gym environment to run training script for.")

    # NAME SETTINGS
    parser.add_argument("--save_folder", type=str, default="checkpoints", 
                        help="Path to save model checkpoints and results.")
    parser.add_argument("--model_name", type=str, default="my_model", 
                        help="Name of the model for saving and loading.")

    # TRAIN SETTINGS
    parser.add_argument("--total_timesteps_to_run", type=int,
                        help="Total number of timesteps to run the training.")
    parser.add_argument("--weight_update_per_save", type=int,
                        help="Number of weight updates before saving the model.")

    # SAVE SETTINGS
    parser.add_argument("--iterations_per_weight_update", type=int,
                        help="Number of iterations before updating weights.")

    # SAVE SETTINGS
    parser.add_argument("--procgen_num_levels", type=int,
                        help="Number of Procgen levels to run Impala algorithm.")

                        # SAVE SETTINGS
    parser.add_argument("--continue_training", type=bool, default=False,
                        help="Do you want to continue training from a existing run?")

    args = parser.parse_args()
    return args

def train_model(env, iterations_per_weight_update: int, weight_update_per_save: int, save_folder: str, 
    model_name: int, total_timesteps_to_run: int, continue_training: str):
    # setup 
    tensorboard_log_dir = os.path.join(save_folder, "logs/")
    writer = SummaryWriter(tensorboard_log_dir)


    # Define checkpointing settings
    checkpoint_callback = CheckpointCallback(
        save_freq = iterations_per_weight_update * weight_update_per_save, # number of timesteps (in env world)
        save_path=save_folder,
        name_prefix=model_name,
    )
    
    # train from scratch
    if continue_training:
        # continue from checkpoint, load latest model
        checkpoint, highest_timestep = get_filename_with_highest_timestep(save_folder)
        total_timesteps_to_run -= highest_timestep
        checkpoint_without_zip = checkpoint.split(".")[-2]
        last_checkpoint = os.path.join(save_folder, checkpoint_without_zip)
        model = PPO.load(last_checkpoint, env=env, device="cuda", tensorboard_log=tensorboard_log_dir)
        model.num_timesteps = highest_timestep
        print(f"Continuing training from '{last_checkpoint}' for {total_timesteps_to_run - highest_timestep} more steps")
    else:
        model = PPO(
            ImpalaActorCriticPolicy,
            env,
            n_steps = iterations_per_weight_update, # timesteps before updating weights
            learning_rate=5e-4,  # 5 × 10^−4
            batch_size=32,  # n_steps / minibatches (256 / 8)
            n_epochs=3,  # Epochs per rollout
            gamma=0.999,  # Discount factor
            gae_lambda=0.95,  # GAE parameter
            ent_coef=0.2,  # Entropy bonus (default 0.03)
            clip_range=0.4,  # PPO clipping range (default 0.2)
            normalize_advantage=True,  # Reward normalization
            device="cuda",
            tensorboard_log=tensorboard_log_dir,
            verbose=1,)

    # Train the PPO model with the callback
    model.learn(
        total_timesteps=total_timesteps_to_run,
        callback=checkpoint_callback,
        log_interval=100,
        tb_log_name=model_name,
        reset_num_timesteps=False
    )

    writer.close()

def evaluation(env, save_folder: str, model_name: str):
    checkpoint_files = [f for f in os.listdir(save_folder) if f.startswith(model_name) and f.endswith("_steps.zip")]

    mean_rewards = []
    std_rewards = []

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(save_folder, checkpoint_file)
        _ = env.reset()

        # run e valuation
        loaded_model = PPO.load(checkpoint_path, env=env)
        mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=10)

        # save for graphing
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)

    timesteps = [int(checkpoint_file.split("_")[1]) for checkpoint_file in checkpoint_files]


    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.errorbar(timesteps, mean_rewards, yerr=std_rewards, fmt='-o', capsize=5, label="Mean Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Model Eval Performance")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save the plot to the save_folder
    plot_path = os.path.join(save_folder, f"{model_name}_eval_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")




if __name__ == "__main__":
    args = parse_args()

    # Create the environment
    # env = gym.make(args.gym_env, start_level=0, num_levels=1)
    print(f"Procgen Levels: {args.procgen_num_levels}")
    env = make_wrapped_procgen_env(args.gym_env, starting_level=0, num_levels=args.procgen_num_levels)
    train_model(env, args.iterations_per_weight_update, 
        args.weight_update_per_save, args.save_folder, 
        args.model_name, args.total_timesteps_to_run,
        args.continue_training)

    # evaluation(env, args.save_folder, args.model_name)


    