import argparse
import json

import torch
import yaml
from datasets import load_dataset
from huggingface_hub import login
from transformer_lens import HookedTransformer

from trainer import Trainer


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


def load_pile_lmsys_mixed_tokens():
    try:
        print("Loading data from disk")
        all_tokens = torch.load(
            "workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
    except Exception:
        print("Data is not cached. Loading data from HF")
        data = load_dataset(
            "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2",
            split="train",
            cache_dir="workspace/cache/",
        )
        data.save_to_disk(
            "workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.hf")
        data.set_format(type="torch", columns=["input_ids"])
        all_tokens = data["input_ids"]
        torch.save(all_tokens,
                   "workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
        print("Saved tokens to disk")
    return all_tokens


with open("config.yaml", "r") as file:
    default_cfg = yaml.safe_load(file)

login(token=default_cfg["hf_token"])
base_model = HookedTransformer.from_pretrained(
    "gemma-2-2b",
    device=default_cfg["device"],
)
chat_model = HookedTransformer.from_pretrained(
    "gemma-2-2b-it",
    device=default_cfg["device"],
)
default_cfg["d_in"] = base_model.cfg.d_model
all_tokens = load_pile_lmsys_mixed_tokens()
cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
