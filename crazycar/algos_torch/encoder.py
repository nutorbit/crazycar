import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from crazycar.agents.constants import SENSOR_SHAPE, CAMERA_HEIGHT, CAMERA_DEPT


class Sensor(nn.Module):
    """
    For sensor feature with 256 feature
    """

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(SENSOR_SHAPE[0], 256)
        )

    def forward(self, obs):
        if isinstance(obs['sensor'], np.ndarray):
            obs['sensor'] = torch.FloatTensor(obs['sensor']).to('cuda')
        x = self.encode(obs['sensor'])
        return x


class Image(nn.Module):
    """
    For image feature with 256 feature
    """

    def __init__(self):
        super().__init__()
        self.encode = ImpalaCNN(CAMERA_HEIGHT, CAMERA_DEPT)

    def forward(self, obs):
        if isinstance(obs['image'], np.ndarray):
            obs['image'] = torch.FloatTensor(obs['image']).to('cuda')
        x = self.encode(obs['image'])
        return x


class Combine(nn.Module):
    """
    Use all feature with 512 feature
    """

    def __init__(self):
        super().__init__()
        self.image_reps = ImpalaCNN(CAMERA_HEIGHT, CAMERA_DEPT)
        self.sensor_reps = nn.Sequential(
            nn.Linear(SENSOR_SHAPE[0], 256)
        )

    def forward(self, obs):
        if isinstance(obs['sensor'], np.ndarray):
            obs['sensor'] = torch.FloatTensor(obs['sensor']).to('cuda')
        if isinstance(obs['image'], np.ndarray):
            obs['image'] = torch.FloatTensor(obs['image']).to('cuda')
        image_reps = self.image_reps(obs['image'])
        sensor_reps = self.sensor_reps(obs['sensor'])
        concat = torch.cat([image_reps, sensor_reps], dim=1)
        return concat


class ImpalaCNN(nn.Module):
    """
    The CNN architecture used in the IMPALA paper.
    Ref: https://arxiv.org/abs/1802.01561
    """

    def __init__(self, image_size, depth_in=4):
        super().__init__()
        layers = []
        for depth_out in [16, 32, 32]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
            depth_in = depth_out
        self.conv_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(math.ceil(image_size / 8) ** 2 * depth_in, 256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        return x


class ImpalaResidual(nn.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, depth):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x