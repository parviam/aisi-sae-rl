from procgen import ProcgenGym3Env
import numpy as np
import gym3
from gym3 import types_np
from PIL import Image as Image
import time
import os
import torch

# This version uses gym3. It works locally but our pipeline does not support it (yet)
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
    observation = observation[0]

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
    states_collected = 0

    while states_collected < num_states:
        states_collected += 1

        env.act(gym3.types_np.sample(env.ac_space, bshape=(env.num,)))
        reward, observation, first = env.observe()

        save_observation(observation['rgb'], states_collected, out_folder)

    env.close()


if __name__ == '__main__':
    env = ProcgenGym3Env(num=1, env_name="coinrun")
    policy = lambda obs: env.action_space.sample()

    # Simple version that does not include feature quotas
    get_states(env, policy, num_states=5, out_folder="./env_states")