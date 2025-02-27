import gymnasium as gym
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from stable_baselines3.common import env_checker

def make_wrapped_procgen_env(gym_id, num_levels=0, starting_level=0, distribution_mode="easy"):
    coin_non_env = GymV21CompatibilityV0("procgen:procgen-coinrun-v0", make_kwargs={
        "num_levels": num_levels,
        "start_level": starting_level,
        "distribution_mode": distribution_mode
    })

    
    # Apply the custom wrapper
    wrapped_env = CustomProcgenEnvWrapper(coin_non_env)

    # stable baselines check for correct environment api
    env_checker.check_env(wrapped_env)
    
    return wrapped_env


# Custom wrapper to ensure compatibility with Stable Baselines3 and the 'seed' parameter
class CustomProcgenEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        # Ignore the seed parameter and just call the original reset method
        if 'seed' in kwargs:
            del kwargs['seed']  # Ignore the seed parameter, but still accept it
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info =  self.env.step(action)
        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)
        return (state, reward, terminated, truncated, info)

    def render(self):
        return self.env.render()

    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
