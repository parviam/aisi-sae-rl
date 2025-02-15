from utils import *
from trainer import Trainer


with open('config.yaml', 'r') as file:
    default_cfg = yaml.safe_load(file)
    
base_model = HookedTransformer.from_pretrained(
    "gemma-2-2b", 
    device=default_cfg['device'], 
)
chat_model = HookedTransformer.from_pretrained(
    "gemma-2-2b-it", 
    device=default_cfg['device'], 
)
default_cfg['d_in'] = base_model.cfg.d_model
all_tokens = load_pile_lmsys_mixed_tokens()
cfg = arg_parse_update_cfg(default_cfg)
trainer = Trainer(cfg, base_model, chat_model, all_tokens)
trainer.train()
