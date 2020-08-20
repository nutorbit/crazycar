import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from crazycar.agents.constants import DISTANCE_SENSORS, CAMERA_HEIGHT
from crazycar.utils import weight_init


class Base(nn.Module):
    """
    Base class for encoder state
    """

    def __init__(self):
        super().__init__()

    def forward(self, obs):
        raise NotImplementedError


class Sensor(Base):
    """
    For sensor feature with 256 feature
    """

    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(len(DISTANCE_SENSORS), 256), nn.ReLU()
        )
        self.out_size = 256

    def forward(self, obs):
        x = self.encode(obs['sensor'])
        return x


class Image(Base):
    """
    For image feature with 256 feature
    """

    def __init__(self):
        super().__init__()
        self.encode = ImpalaCNN(CAMERA_HEIGHT)
        self.out_size = 256

    def forward(self, obs):
        x = self.encode(obs['image'])
        return x


class Combine(Base):
    """
    Use all feature with 512 feature
    """

    def __init__(self):
        super().__init__()
        self.image_reps = ImpalaCNN(CAMERA_HEIGHT)
        self.sensor_reps = nn.Sequential(
            nn.Linear(len(DISTANCE_SENSORS), 256),
            nn.ReLU()
        )
        self.out_size = 512

    def forward(self, obs):
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

        self.apply(weight_init)

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

        self.apply(weight_init)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x