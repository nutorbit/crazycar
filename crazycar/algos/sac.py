import torch
import torch.nn as nn

from copy import deepcopy
from torch.optim import Adam

from crazycar.utils import make_mlp, weight_init
from crazycar.algos.model import BaseModel


class Actor(BaseModel):
    def __int__(self, encoder, hiddens=[256, 256]):
        super().__int__()
        self.enc = deepcopy(encoder)


class Critic(BaseModel):
    def __init__(self, encoder, hiddens=[256, 256]):
        super().__init__()
        self.enc = deepcopy(encoder)


class SAC:
    """
    Soft Actor-Critic

    Args:
        ...
    """

    def __init__(self, encoder):
        pass
