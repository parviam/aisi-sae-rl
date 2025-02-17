import gymnasium as gym
import torch
import tqdm
import numpy as np
import einops


class GameStateBuffer:
    """Modified Buffer class to store game states from procgen environments"""

    def __init__(self, cfg, model_A, model_B, all_tokens, env_name="procgen:procgen-coinrun-v0"):
        self.cfg = cfg
        self.buffer_size = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = self.buffer_size // (cfg["seq_len"] - 1)
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1)
        self.model_A = model_A
        self.model_B = model_B
        self.pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens

        # Initialize buffer to store game states
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model), # hardcoding 2 for model diffing
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"])

        # Create environment
        self.env = gym.make(env_name)
        self.policy = lambda obs: self.env.action_space.sample()  # Random policy

        # Feature detection settings (TODO)
        self.feature_color_ranges = {
            'coin': {'color_range': [(200, 200, 0), (255, 255, 100)]},
            'enemy': {'color_range': [(150, 0, 150), (255, 100, 255)]},
            'box': {'color_range': [(139, 69, 19), (160, 82, 45)]},
            'lava': {'color_range': [(255, 69, 0), (255, 140, 0)]}
        }
        self.feature_counts = {feature: 0 for feature in self.feature_color_ranges}

        # Initialize normalization factors
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)

        self.normalisation_factor = torch.tensor(
            [
                estimated_norm_scaling_factor_A,
                estimated_norm_scaling_factor_B,
            ], device="cuda:0",dtype=torch.float32,)

        self.refresh()

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


    def has_feature(self, observation):
        """Check if an observation contains specific features"""
        features_present = {}
        for feature, ranges in self.feature_color_ranges.items():
            color_range = ranges['color_range']
            pixels_in_range = np.any((observation >= color_range[0]) &
                                     (observation <= color_range[1]).all(axis=-1))
            features_present[feature] = pixels_in_range
        return features_present

    @torch.no_grad()
    def refresh(self):
        """Collect new diverse game states to refresh the buffer"""
        self.pointer = 0
        print("Refreshing the buffer with new game states!")

        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_states = self.buffer_batches
            else:
                num_states = self.buffer_batches // 2
            self.first = False

            states_collected = 0
            observation, _ = self.env.reset()

            for _ in tqdm.trange(num_states):
                # Check features in current state
                features = self.has_feature(observation)

                # Determine if we should save this state
                #should_save = any(present for present in features.values()) or np.random.random() < 0.2
                should_save = True #(TODO) Temporarily bypassing feature quotas.

                if should_save and states_collected < num_states:
                    # Process the observation through both models
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(observation).to(self.cfg["device"])
                        acts_A = self.model_A(obs_tensor)
                        acts_B = self.model_B(obs_tensor)

                        # Stack activations
                        acts = torch.stack([acts_A, acts_B], dim=0)

                        # Store in buffer
                        self.buffer[states_collected] = acts
                        states_collected += 1

                # Take action and get next state
                action = self.policy(observation)
                observation, _, terminated, truncated, _ = self.env.step(action)

                if terminated or truncated:
                    observation, _ = self.env.reset()

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