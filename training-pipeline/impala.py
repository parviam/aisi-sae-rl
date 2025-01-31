import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from gym.spaces import Box


class Config:
    USE_BATCH_NORM = 1
    DROPOUT = 0
    ARCHITECTURE = 'impala'

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, use_batch_norm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)
            
        self.dropout = nn.Dropout2d(p=Config.DROPOUT) if Config.DROPOUT > 0 else None

    def forward(self, x):
        out = F.relu(x)
        
        out = self.conv1(out)
        if self.dropout:
            out = self.dropout(out)
        if self.use_batch_norm:
            out = self.bn1(out)
            
        out = F.relu(out)
        
        out = self.conv2(out)
        if self.dropout:
            out = self.dropout(out)
        if self.use_batch_norm:
            out = self.bn2(out)
            
        return out + x

class ImpalaCNN(nn.Module):
    def __init__(self, num_channels=3, depths=[16, 32, 32]):
        super().__init__()
        self.use_batch_norm = Config.USE_BATCH_NORM == 1
        
        layers = []
        in_channels = num_channels
        
        for depth in depths:
            layers.extend([
                nn.Conv2d(in_channels, depth, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaBlock(depth, self.use_batch_norm),
                ImpalaBlock(depth, self.use_batch_norm)
            ])
            in_channels = depth
            
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(self._get_conv_output_size(num_channels), 256)
        
    def _get_conv_output_size(self, channels):
        x = torch.zeros(1, channels, 64, 64)
        x = self.conv_layers(x)
        return int(np.prod(x.shape[1:]))
        
    def forward(self, images):
        x = images.float() / 255.0
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(x)
        x = F.relu(self.fc(x))
        return x

class ImpalaPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        self.cnn = ImpalaCNN(num_channels=observation_space.shape[0])
        self.actor = nn.Linear(256, action_space.n)
        self.critic = nn.Linear(256, 1)

    def forward(self, obs, deterministic=False):
        features = self.cnn(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        
        action_probs = torch.softmax(action_logits, dim=-1)
        distribution = torch.distributions.Categorical(action_probs)
        
        actions = None
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        
        return actions, value, log_prob