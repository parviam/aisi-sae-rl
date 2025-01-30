import gym
import procgen
import numpy as np
import pygame
from PIL import Image as PILImage
import time
import torch


def run_environment(env: gym.Env, num_episodes: int = 1, render_with_pygame: bool = True, policy: torch.nn.Module = None):
    """
    Main function to run the environment.

    Args:
        env (gym.Env): The Procgen environment
        num_episodes (int): Number of episodes to run. Default 1.
        render_with_pygame (bool): Whether to render the environment using pygame. Default True.
        policy (nn module): Pretrained policy. If left as "None", the environment will run with a random policy.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(num_episodes):
        if render_with_pygame:
            pygame.init()
            display_surface = pygame.display.set_mode((512, 512), 0, 32)
            clock = pygame.time.Clock()
            pygame.display.flip()

        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            if policy is not None:
                state = torch.FloatTensor(obs).unsqueeze(0).permute(0, 3, 1, 2).to(device) # Process observation so it can be provided to the policy
                with torch.no_grad():
                    action_probs, _ = policy(state)
                    action = torch.argmax(action_probs).item()

            else: # If no policy provided, run random policy
                action = env.action_space.sample()

            obs, reward, done, info = env.step(action)

            if render_with_pygame:
                image = env.render(mode='rgb_array')
                if image is not None:
                    image = PILImage.fromarray(image)
                    pygame_surface = pygame.image.fromstring(
                        image.tobytes(), image.size, image.mode)
                    display_surface.blit(pygame_surface, (0, 0))
                    pygame.display.flip()
                    clock.tick(144)  # FPS limiter

            total_reward += reward

        print(f"Episode {episode + 1} completed with total reward: {total_reward}")

def make_and_run_environment(env_name: str = "coinrun", num_levels: int = 10, start_level: int = 0, distribution_mode: str = "easy",  render_mode: str = None, num_episodes: int = 1, render_with_pygame: bool = True, policy: torch.nn.Module = None):
    """
    Wrapper method that makes a given procgen environment, runs a model on the environemnt, and exits gracefully.

    Args:
        env_name (string): The name of the Procgen environment to run. Default \"coinrun\".
        num_levels (int): The number of levels for the env. Default 10.
        start_level (int): What level to start at. Default 0.
        distribution_mode (str): Difficulty of levels generated. "easy" (default), "hard", "extreme", or "memory".
        render_mode (str): Determines how env should be visualized. "human", "rgb_array", None (default) are common.
        num_episodes (int): Number of episodes to run. Default 1.
        render (bool): Whether to render the environment using pygame. Default True.
        policy (nn module): Pretrained policy. If left as "None", the environment will run with a random policy.
    """
    full_env_name = f"procgen:procgen-{env_name}-v0"
    env = gym.make(full_env_name, num_levels=num_levels, start_level=start_level,distribution_mode=distribution_mode, render_mode=render_mode)
    run_environment(env, num_episodes=num_episodes, render_with_pygame=render_with_pygame, policy=policy)
    env.close()

if __name__ == "__main__":
    make_and_run_environment(env_name="coinrun", num_levels=10, start_level=0, distribution_mode="easy", render_mode="rgb_array")