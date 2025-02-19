import gymnasium as gym
from procgen import ProcgenGym3Env
import numpy as np
import gym3
import pygame
from gym3 import types_np
from PIL import Image as Image
import time
import os
import torch




######################## Simple implementation - Just saves n states from coinrun ##########################

def save_observation(observation, step, out_folder):
    """
    Helper function to save an observation as an image file.

    Args:
        observation (numpy.ndarray): The environment observation with shape (1,64,64,3)
        step (int): The current step number (used for filename)
        out_folder (str): Directory to save the image
    """
    # Remove the batch dimension to get (64,64,3)
    observation = observation[0]  # Get the first (and only) item from the batch

    # Ensure the pixel values are in the correct range for PNG (0-255)
    if observation.dtype == np.float32 or observation.dtype == np.float64:
        observation = (observation * 255).astype(np.uint8)

    img = Image.fromarray(observation)
    filename = os.path.join(out_folder, f'state_{step:05d}.png')
    img.save(filename)


def get_states(env, policy, num_states, out_folder):
    """
    Play through an environment using a given policy and save observations.

    Args:
        env: Gymnasium environment (image-based like procgen)
        policy: Policy function that takes observation and returns action
        num_states (int): Number of states to collect
        out_folder (str): Directory to save the observations
    """
    os.makedirs(out_folder, exist_ok=True)

    #observation, info = env.reset()

    states_collected = 0
    while states_collected < num_states:

        action = types_np.sample(env.ac_space, bshape=(env.num,))
        env.act(action)
        rew, obs, first = env.observe()
        print(f"step {states_collected}, action {action}  reward {rew} first {first}")

        if states_collected > 0 and first:
            break
        states_collected += 1


        save_observation(obs['rgb'], states_collected, out_folder)

        # Old format (doesn't work for gym3 procgen env
        #action = policy(observation)
        #observation, reward, terminated, truncated, info = env.step(action)
        #if terminated or truncated:
            #observation, info = env.reset()

    env.close()

########################### More advanced version with feature quotas ##################################################

def has_feature(observation, feature_color_ranges):
    """
    Check if an observation contains a specific feature based on color ranges.

    Args:
        observation: numpy array of the game state
        feature_color_ranges: dict of color ranges for different features
    Returns:
        dict: Boolean indicators for each feature's presence
    """
    features_present = {}
    for feature, color_range in feature_color_ranges.items():
        # Check if any pixels fall within the color range for this feature
        pixels_in_range = np.any((observation >= color_range[0]) &
                                 (observation <= color_range[1]).all(axis=-1))
        features_present[feature] = pixels_in_range
    return features_present


def get_diverse_states(env, policy, num_states, out_folder, min_feature_counts=None):
    """
    Collect diverse game states ensuring representation of key features.

    Args:
        env: Gymnasium environment
        policy: Policy function that takes observation and returns action
        num_states: Total number of states to collect
        out_folder: Directory to save observations
        min_feature_counts: Dict specifying minimum number of states needed for each feature
    """
    os.makedirs(out_folder, exist_ok=True)

    # Define color ranges for key features
    feature_color_ranges = {
        'coin': {'color_range': [(200, 200, 0), (255, 255, 100)]},  # Gold coins
        'enemy': {'color_range': [(150, 0, 150), (255, 100, 255)]},  # Purple enemies
        'box': {'color_range': [(139, 69, 19), (160, 82, 45)]},  # Brown boxes
        'lava': {'color_range': [(255, 69, 0), (255, 140, 0)]}  # Orange lava
    }

    # Initialize feature counts
    feature_counts = {feature: 0 for feature in feature_color_ranges}

    # Set minimum counts if not provided
    if min_feature_counts is None:
        min_feature_counts = {feature: num_states // len(feature_color_ranges)
                              for feature in feature_color_ranges}

    observation, info = env.reset()
    states_collected = 0
    consecutive_no_new_features = 0
    max_attempts = num_states * 10  # Prevent infinite loops
    attempts = 0

    while states_collected < num_states and attempts < max_attempts:
        features = has_feature(observation, feature_color_ranges)

        # Save state if it has any underrepresented features
        should_save = False
        for feature, present in features.items():
            if present and feature_counts[feature] < min_feature_counts[feature]:
                should_save = True
                break

        # Also save some random states to maintain variety
        if not should_save and np.random.random() < 0.2:  # 20% chance
            should_save = True

        if should_save and states_collected < num_states:
            # Save the observation
            img = Image.fromarray(observation)
            filename = os.path.join(out_folder, f'state_{states_collected:05d}.png')
            img.save(filename)

            # Update feature counts
            for feature, present in features.items():
                if present:
                    feature_counts[feature] += 1

            states_collected += 1

            # Log progress
            if states_collected % 100 == 0:
                print(f"Collected {states_collected} states. Feature counts:")
                for feature, count in feature_counts.items():
                    print(f"{feature}: {count}/{min_feature_counts[feature]}")

        # Take action and get next state
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        attempts += 1

    env.close()

    print("\nFinal feature counts:")
    for feature, count in feature_counts.items():
        print(f"{feature}: {count}/{min_feature_counts[feature]}")

    return feature_counts

if __name__ == '__main__':

    #env = gym.make("procgen-coinrun-v0") # Deprecated
    env = ProcgenGym3Env(num=1, env_name="coinrun",render_mode="rgb_array")

    policy = lambda obs: env.action_space.sample()

    # Simple version that does not include feature quotas
    get_states(env, policy, num_states=5, out_folder="./saved_states")

    # More complex version that sets quotas for each feature type (not fully implemented yet)
    min_counts = {
        'coin': 150,
        'enemy': 100,
        'box': 100,
        'lava': 50
    }
    #feature_counts = get_diverse_states(env, policy, num_states=500, out_folder="./diverse_states", min_feature_counts=min_counts)