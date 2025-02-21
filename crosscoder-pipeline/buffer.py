import gym
import torch
import tqdm
import numpy as np
from PIL import Image
import glob
import einops
import time
import os
# from procgen import ProcgenGym3Env
# import gym3 # For gym3 version of coinrun


class Buffer:
    """Modified Buffer class to store game states from procgen environments.

    Environment states can be preloaded to /env_states/ and loaded here if self.states_provided == True, or generated ad hoc in the refresh() method if states_provided == False.

    State and action pairs are also logged to an NPZ file which contains a dictionary of lists:
        state_actions = {
                         states (list of arrays)
                         activations_A (list of arrays)
                         activations_B (list of arrays)
                         }

    """

    def __init__(self, cfg, model_A, model_B, all_tokens, env_name="procgen:procgen-coinrun-v0",verbose = False):
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.states_provided = cfg["states_provided"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.model_A = model_A
        self.model_B = model_B
        self.pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        self.verbose = verbose

        # Initialize buffer to store game states
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model), # hardcoding 2 for model diffing
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])

        # Create environment
        self.env = gym.make(env_name)
        self.policy = lambda observation: self.env.action_space.sample()  # Random policy

        self.state_activation_pairs = {
            'states': [],  # Will store numpy arrays of shape (64, 64, 3)
            'activations_A': [],  # Will store activations from model A
            'activations_B': []  # Will store activations from model B
        }

        # Initialize normalization factors
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)

        self.normalisation_factor = torch.tensor(
            [estimated_norm_scaling_factor_A, estimated_norm_scaling_factor_B,],
            device=cfg["device"], dtype=torch.float32,
        )

        if self.states_provided:
            self.state_files = glob.glob(os.path.join("env_states", "*.png"))
            if len(self.state_files) == 0:
                raise ValueError("No PNG files found in env_states folder")
            if self.verbose:
                print(f"Found {len(self.state_files)} state files in env_states folder")

        self.refresh()

    def save_state_activation_pairs(self, filename, accumulate=True):
        """Save the collected state-activation pairs to a file

        Args:
            filename: The file to save to
            accumulate: If True and the file exists, append to existing data
        """
        new_data = {
            'states': np.array(self.state_activation_pairs['states']),
            'activations_A': np.array([act.cpu().numpy() for act in self.state_activation_pairs['activations_A']]),
            'activations_B': np.array([act.cpu().numpy() for act in self.state_activation_pairs['activations_B']])
        }

        if accumulate and os.path.exists(filename):
            # Load existing data
            existing_data = np.load(filename)
            # Concatenate with new data
            save_dict = {
                'states': np.concatenate([existing_data['states'], new_data['states']]),
                'activations_A': np.concatenate([existing_data['activations_A'], new_data['activations_A']]),
                'activations_B': np.concatenate([existing_data['activations_B'], new_data['activations_B']])
            }
            if self.verbose: print(f"Added {len(new_data['states'])} new samples to existing {len(existing_data['states'])} samples")
        else:
            save_dict = new_data
            if self.verbose: print(f"Saved {len(new_data['states'])} samples to new file")

        np.savez_compressed(filename, **save_dict)


    def load_state_from_png(self, filepath):
        """Load and process a state from a PNG file
        Args:
            filepath: Path to the PNG file
        Returns:
            numpy array of shape (64, 64, 3)
        """
        img = Image.open(filepath)
        state = np.array(img, dtype=np.uint8) # Convert back to numpy array

        if state.shape != (64, 64, 3):
            raise ValueError(f"Invalid state shape: {state.shape}, expected (64, 64, 3)")
        return state


    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 100):
        norms_per_batch = []
        for i in tqdm.tqdm(range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"):
            tokens = self.all_tokens[i * batch_size: (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm
        return scaling_factor


    @torch.no_grad()
    def refresh(self):
        """Collect new diverse game states to refresh the buffer"""
        self.pointer = 0
        if self.verbose: print("Refreshing the buffer with new game states!")

        #with torch.autocast("cuda", torch.bfloat16):
        with torch.no_grad():
            if self.first:
                num_states = self.buffer_batches
            else:
                num_states = self.buffer_batches // 2
            self.first = False

            states_collected = 0

            for _ in tqdm.trange(num_states):
                with torch.no_grad():
                    if self.states_provided:
                        state_file = self.state_files[0]
                        observation = self.load_state_from_png(state_file)

                    else: # Sample observations from env
                        if states_collected == 0:
                            observation = self.env.reset()
                        action = self.policy(observation)
                        observation, reward, done, info = self.env.step(action)

                    obs_tensor = torch.from_numpy(observation).to(self.cfg["device"])

                    # Process the observation through both models
                    acts_A = self.model_A(obs_tensor)
                    acts_B = self.model_B(obs_tensor)

                    # Store states and activations for later correlation
                    self.state_activation_pairs['states'].append(observation.copy())
                    self.state_activation_pairs['activations_A'].append(acts_A.detach().clone())
                    self.state_activation_pairs['activations_B'].append(acts_B.detach().clone())
                    self.save_state_activation_pairs("test_state_acts.npz")

                    # Stack activations
                    acts = torch.stack([acts_A, acts_B], dim=0)

                    # Store in buffer
                    self.buffer[self.pointer: self.pointer + acts.shape[0]] = acts
                    self.pointer += acts.shape[0]
                    states_collected += 1

        # Shuffle buffer
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])]
        self.pointer = 0


    @torch.no_grad()
    def next(self):
        """Get next batch of states"""
        out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]].float()
        self.pointer += self.cfg["batch_size"]

        if self.pointer > self.buffer.shape[0] // 2 - self.cfg["batch_size"]:
            self.refresh()

        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out


####### Content below is for unit testing #######

class MockModel:
    """Mock model class for testing"""

    def __init__(self, d_model=64):
        self.cfg = type('Config', (), {'d_model': d_model})()

    def __call__(self, x):
        # Return a constant tensor with the same batch size as input
        return torch.ones(self.cfg.d_model, device=x.device, dtype=torch.float32)

    def run_with_cache(self, tokens, names_filter=None, return_type=None):
        batch_size = len(tokens)
        cache = {names_filter: torch.ones(self.cfg.d_model, device=tokens.device)}
        return None, cache


def main(): # For testing
    test_cfg = {
        "batch_size": 32,
        "buffer_mult": 4,
        "seq_len": 9,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "model_batch_size": 16,
        "hook_point": "hook_point"
    }

    model_A = MockModel()
    model_B = MockModel()

    # Create mock tokens
    all_tokens = torch.randint(0, 100, (1000,))

    try:
        buffer = Buffer(test_cfg, model_A, model_B, all_tokens)

        batch = buffer.next()
        print("\nTest Results:")
        print(f"Batch shape: {batch.shape}")
        print(f"Expected shape: torch.Size([{test_cfg['batch_size']}, 2, {model_A.cfg.d_model}])")
        print(f"Buffer pointer position: {buffer.pointer}")
        print("Buffer test completed successfully!")

    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")

if __name__ == "__main__":
    main()