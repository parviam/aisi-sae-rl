import warnings
warnings.filterwarnings('ignore')

import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import tqdm
import numpy as np
from PIL import Image
import glob
import einops
import time
import os

from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
# from procgen import ProcgenGym3Env


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

    def __init__(self, cfg, model_A, model_B, env_name="coinrun",verbose = False):
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
        self.verbose = verbose

        # Initialize buffer to store game states
        self.buffer = np.zeros(
            (self.buffer_size, 2, cfg['d_model']), # hardcoding 2 for model diffing
            dtype=np.float16,
        )

        # Create environment
        venv = ProcgenEnv(num_envs=1,
                      env_name=env_name,
                      num_levels=1000,
                      start_level=0,
                      distribution_mode='hard')
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        self.env = VecNormalize(venv=venv, ob=False)

        self.policy = lambda observation: self.env.action_space.sample()  # Random policy

        self.state_activation_pairs = {
            'states': [],  # Will store numpy arrays of shape (64, 64, 3)
            'activations_A': [],  # Will store activations from model A
            'activations_B': []  # Will store activations from model B
        }

        if self.states_provided:
            self.state_files = glob.glob(os.path.join("env_states", "*.png"))
            if len(self.state_files) == 0:
                raise ValueError("No PNG files found in env_states folder")
            if self.verbose:
                print(f"Found {len(self.state_files)} state files in env_states folder")

        self.refresh()
        
        # Initialize normalization factors
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)

        self.normalisation_factor = tf.constant(
            [estimated_norm_scaling_factor_A, estimated_norm_scaling_factor_B,],
            dtype=tf.float32,
        )

    def save_state_activation_pairs(self, filename, accumulate=True):
        """Save the collected state-activation pairs to a file

        Args:
            filename: The file to save to
            accumulate: If True and the file exists, append to existing data
        """
        new_data = {
            'states': np.array(self.state_activation_pairs['states']),
            'activations_A': np.array(self.state_activation_pairs['activations_A']),
            'activations_B': np.array(self.state_activation_pairs['activations_B'])
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

        # import pdb;pdb.set_trace()
        # np.savez_compressed(filename, **save_dict)


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
            states = self.state_activation_pairs['states'][i * batch_size: (i + 1) * batch_size]
            states = tf.convert_to_tensor(np.concatenate(states))
            states = tf.cast(states, tf.float32)
            acts = model(states)
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            acts = acts.eval(session=sess)
            norms_per_batch.append(np.mean(np.linalg.norm(acts, axis=-1)))
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.cfg['d_model']) / mean_norm
        return scaling_factor


    def refresh(self):
        """Collect new diverse game states to refresh the buffer"""
        self.pointer = 0
        if self.verbose: print("Refreshing the buffer with new game states!")

        if self.first:
            num_states = self.buffer_batches
        else:
            num_states = self.buffer_batches // 2
        self.first = False

        states_collected = 0

        for _ in tqdm.trange(num_states):
            if self.states_provided:
                state_file = self.state_files[0]
                observation = self.load_state_from_png(state_file)

            else: # Sample observations from env
                if states_collected == 0:
                    observation = self.env.reset()
                action = np.array(self.policy(observation))
                observation, reward, done, info = self.env.step(action)

            obs_tensor = tf.convert_to_tensor(observation)
            obs_tensor = tf.cast(obs_tensor, tf.float32)
            # obs_tensor = tf.reshape(obs_tensor, (1,3,64,64))

            # Process the observation through both models
            acts_A = self.model_A(obs_tensor)
            acts_B = self.model_B(obs_tensor)
            
            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)
            acts_A = acts_A.eval(session=sess)
            acts_B = acts_B.eval(session=sess)

            # Store states and activations for later correlation
            self.state_activation_pairs['states'].append(observation)
            self.state_activation_pairs['activations_A'].append(acts_A)
            self.state_activation_pairs['activations_B'].append(acts_B)
            self.save_state_activation_pairs("test_state_acts.npz")

            # Stack activations
            acts = np.stack([acts_A, acts_B], axis=0)

            # Store in buffer
            self.buffer[self.pointer: self.pointer + acts.shape[0]] = acts
            self.pointer += acts.shape[0]
            states_collected += 1

        # Shuffle buffer
        self.buffer = np.random.shuffle(self.buffer)
        self.pointer = 0


    def next(self):
        """Get next batch of states"""
        out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
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
        return tf.ones(self.cfg['d_model'], dtype=tf.float32)

    def run_with_cache(self, states, names_filter=None, return_type=None):
        batch_size = len(states)
        cache = {names_filter: tf.ones(self.cfg['d_model'], dtype=tf.float32)}
        return None, cache


def main(): # For testing
    test_cfg = {
        "batch_size": 32,
        "buffer_mult": 4,
        "seq_len": 9,
        "device": "cuda:0" if len(tf.config.list_physical_devices('GPU')) > 0 else "cpu",
        "model_batch_size": 16,
        "hook_point": "hook_point",
        "states_provided": False
    }

    model_A = MockModel()
    model_B = MockModel()


    try:
        buffer = Buffer(test_cfg, model_A, model_B)

        batch = buffer.next()
        print("\nTest Results:")
        print(f"Batch shape: {batch.shape}")
        print(f"Expected shape: [{test_cfg['batch_size']}, 2, {self.cfg['d_model']}]")
        print(f"Buffer pointer position: {buffer.pointer}")
        print("Buffer test completed successfully!")

    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")

if __name__ == "__main__":
    main()
