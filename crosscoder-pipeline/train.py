import argparse
import json
import sys

import torch
import yaml
from datasets import load_dataset
import gym
from huggingface_hub import login
from stable_baselines3 import PPO
from transformer_lens import HookedTransformer

from trainer import Trainer

sys.path.insert(1, '../training-pipeline')
from impala import ImpalaPolicy


def arg_parse_update_cfg(default_cfg):
    """
    Helper function to take in a dictionary of arguments,
    convert these to command line arguments,
    look at what was passed in, and return an updated dictionary.
    If in Ipython, just returns with no changes
    """
    cfg = dict(default_cfg)
    parser = argparse.ArgumentParser()
    for key, value in default_cfg.items():
        if type(value) is bool:
            # argparse for Booleans is broken rip.
            # Now you put in a flag to change the default --{flag} to set True,
            # --{flag} to set False
            if value:
                parser.add_argument(f"--{key}", action="store_false")
            else:
                parser.add_argument(f"--{key}", action="store_true")
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args()
    parsed_args = vars(args)
    cfg.update(parsed_args)
    print("Updated config")
    print(json.dumps(cfg, indent=2))
    return cfg


with open("config.yaml", "r") as file:
    default_cfg = yaml.safe_load(file)

login(token=default_cfg["hf_token"])
env = gym.make(f"procgen:procgen-coinrun-v0", render_mode="rgb_array")
model_A = PPO(
        ImpalaPolicy,
        env,
        learning_rate=default_cfg["lr"],
        n_steps = default_cfg["iterations_per_weight_update"], # timesteps before updating weights
        verbose=1,
        device=default_cfg["device"])
model_B = PPO(
        ImpalaPolicy,
        env,
        learning_rate=default_cfg["lr"],
        n_steps = default_cfg["iterations_per_weight_update"], # timesteps before updating weights
        verbose=1,
        device=default_cfg["device"])

# default_cfg["d_in"] = model_A.cfg.d_model
cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, model_A, model_B)
trainer.train()
