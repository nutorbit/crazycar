import numpy as np
import random
import torch
import os
import json

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from cpprb import PrioritizedReplayBuffer as prb
from cpprb import ReplayBuffer as rb


def set_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def huber_loss(x, delta=10.):
    """
    Compute the huber loss.
    Ref: https://en.wikipedia.org/wiki/Huber_loss
    """

    delta = torch.ones_like(x) * delta
    less_than_max = 0.5 * (x * x)
    greater_than_max = delta * (torch.abs(x) - 0.5 * delta)

    return torch.where(
        torch.abs(x) <= delta,
        less_than_max,
        greater_than_max
    )


set_seed(100)


class BaseReplayBuffer:

    def __init__(self, obs_dim, act_dim, size=10000):
        self.default_env_dict = {
            "obs": {
                "shape": obs_dim
            },
            "act": {
                "shape": act_dim
            },
            "rew": {},
            "next_obs": {
                "shape": obs_dim
            },
            "done": {}
        }

    def add(self, obs, next_obs, act, rew, done):
        self.buffer.add(
            obs=np.asarray(obs, dtype=np.float32),
            next_obs=np.asarray(next_obs, dtype=np.float32),
            act=np.asarray(act, dtype=np.float32),
            rew=np.asarray(rew, dtype=np.float32),
            done=np.asarray(done, dtype=np.float32),
        )


class ReplayBuffer(BaseReplayBuffer):

    def __init__(self, obs_dim, act_dim, size=100000):
        super().__init__(obs_dim, act_dim)
        self.buffer = rb(size, env_dict=self.default_env_dict)

    def sample(self, batch_size=32):
        batch = self.buffer.sample(batch_size)
        payload = {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k, v in batch.items()}

        return payload


class PriorityReplayBuffer(BaseReplayBuffer):

    def __init__(self, obs_dim, act_dim, size):
        super().__init__(obs_dim, act_dim)
        self.buffer = prb(size, env_dict=self.default_env_dict)

    def sample(self, batch_size=32, beta=0.4):
        batch = self.buffer.sample(batch_size=batch_size, beta=beta)
        indexes, weights = batch['indexes'], batch['weights']
        # self.buffer.update_priorities(indexes, weights)

        payload = {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k, v in list(batch.items())[:-2]}

        return payload


class Logger:

    def __init__(self):

        self.time = datetime.now()
        self.start_date = self.time.strftime("%b_%d_%Y_%H%M%S")
        self.steps = 0
        self.save_hy = False
        self.writer = None
        self.hyperparameter = None

    def start(self):

        # Create lops directory
        if not os.path.exists(f'./logs/{self.start_date}'):
            os.makedirs(f'./logs/{self.start_date}')

        self.writer = SummaryWriter(f'./logs/{self.start_date}/')

        # Create model directory
        if not os.path.exists(f'./models/{self.start_date}'):
            os.makedirs(f'./models/{self.start_date}')

        with open(f'./logs/{self.start_date}/params.json', 'w') as f:
            json.dump(self.hyperparameter, f)

    def save_hyperparameter(self, **kwargs):

        # Save hyperparameter
        self.hyperparameter = kwargs

    def update_steps(self):
        self.steps += 1

    def save_model(self, model):
        torch.save(model, f'./models/{self.start_date}/td3_{self.steps + 1}.pth')

    def store(self, name, val):
        self.writer.add_scalar(name, val, self.steps)

