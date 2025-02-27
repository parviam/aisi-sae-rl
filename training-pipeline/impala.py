import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    USE_BATCH_NORM = int(os.getenv("USE_BATCH_NORM"))
    DROPOUT = float(os.getenv("DROPOUT"))

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding="same")

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
            
        return out + x

class ImpalaCNN(nn.Module):
    def __init__(self, num_channels=3, features_dim=256, depths=[16, 32, 32]):
        super().__init__()
        
        layers = []
        in_channels = num_channels
        
        for depth in depths:
            layers.extend([
                nn.Conv2d(in_channels, depth, 3, padding="same"),
                nn.MaxPool2d(3, stride=2, padding="same"),
                ImpalaBlock(depth),
                ImpalaBlock(depth)
            ])
            in_channels = depth
            
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(self._get_conv_output_size(num_channels), features_dim)
        self.features_dim = features_dim
        
    def _get_conv_output_size(self, channels):
        x = torch.zeros(1, channels, 64, 64)
        x = self.conv_layers(x)
        return int(np.prod(x.shape[1:]))
        
    def forward(self, images):
        # note - below line not needed when using "normalize_images" in
        # ActorCriticPolicy params
        # x = images.float() / 255.0
        x = self.conv_layers(x)
        x = torch.flatten(x)
        x = F.relu(x)
        x = F.relu(self.fc(x))
        return x

class ImpalaActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, ob_space, ac_space, *args, **kwargs):
        kwargs["ortho_init"] = False
        kwargs["net_arch"] = dict(pi=[256], vf=[256])
        kwargs["activation_fn"] = torch.nn.ReLU
        kwargs["share_features_extractor"] = True
        kwargs["normalize_images"] = True

        super().__init__(
            ob_space,
            ac_space,
            *args,
            **kwargs
        )
        self.ob_space = ob_space
        self.ac_space = ac_space

    def make_feature_extractor(self):
        return ImpalaCNN(num_channels=3, features_dim=256)