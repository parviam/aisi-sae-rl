import argparse
import os

import gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

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

    args = parser.parse_args()
    return args

def train_model(env, iterations_per_weight_update: int, weight_update_per_save: int, save_folder: str, model_name: int, total_timesteps_to_run: int):
    # Define checkpointing settings
    checkpoint_callback = CheckpointCallback(
        save_freq = iterations_per_weight_update * weight_update_per_save, # number of timesteps (in env world)
        save_path=save_folder,
        name_prefix=model_name,
    )

    model = PPO(
        "CnnPolicy", 
        env,
        n_steps = iterations_per_weight_update, # timesteps before updating weights
        verbose=1)

    # Train the PPO model with the callback
    model.learn(
        total_timesteps=total_timesteps_to_run,
        callback=checkpoint_callback,
        log_interval=10
    )

def evaluation(env, save_folder: str, model_name: str):
    latest_checkpoint = f"{save_folder}{model_name}_steps.zip"
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
    env = gym.make(args.gym_env)

    train_model(env, args.iterations_per_weight_update, args.weight_update_per_save, args.save_folder, args.model_name, args.total_timesteps_to_run)

    evaluation(env, args.save_folder, args.model_name)


    