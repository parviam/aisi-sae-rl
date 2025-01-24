import gym
import procgen
import numpy as np
import pygame
from PIL import Image as PILImage
import time
import torch


def run_environment(env, num_episodes=1, render=True, policy=None):
    """
    Main function to run the environment.

    Args:
        env (gym.Env): The Procgen environment
        num_episodes (int): Number of episodes to run
        render (bool): Whether to render the environment using pygame
        policy (nn module): Pretrained policy. If left as "None", the environment will run with a random policy.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(num_episodes):
        if render:
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

            if render:
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


if __name__ == "__main__":

    env_name = "coinrun"
    num_levels = 10
    start_level = 0
    distribution_mode = "easy"
    render_mode = "rgb_array"

    env = gym.make(f"procgen:procgen-{env_name}-v0", num_levels=num_levels, start_level=start_level,distribution_mode=distribution_mode, render_mode=render_mode)

    run_environment(env, num_episodes=2)
    env.close()