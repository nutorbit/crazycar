import numpy as np
import random
import torch
import os
import json

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(100)


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size=100000):
        self.obs = np.zeros((size, *obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act = np.zeros((size, *act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.cap = size

    def add(self, obs, next_obs, act, rew, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size=32):
        idx = np.random.randint(low=0, high=self.size, size=batch_size)
        batch = dict(
            obs=self.obs[idx],
            next_obs=self.next_obs[idx],
            act=self.act[idx],
            rew=self.rew[idx],
            done=self.done[idx]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k, v in batch.items()}


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

