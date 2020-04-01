import numpy as np
import random
import torch
import torch.nn as nn
import os
import json

from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def set_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False


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


# set_seed(100)


def get_default_rb_dict(obs_dim, act_dim, size):
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
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
            "done": {},
        }
    }


class Logger:

    def __init__(self):

        self.time = datetime.now()
        self.start_date = self.time.strftime("%b_%d_%Y_%H%M%S")
        self.steps = 0
        self.writer = None
        self.hyperparameter = None

    def start(self):

        # Create logs directory
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

