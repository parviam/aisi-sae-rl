import os
from stable_baselines3.common.callbacks import BaseCallback

# Custom callback to save checkpoints
class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, model_name: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name
        self.verbose = verbose
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"{self.model_name}_checkpoint_{self.n_calls}.zip")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"Checkpoint saved: {checkpoint_file}")
        return True