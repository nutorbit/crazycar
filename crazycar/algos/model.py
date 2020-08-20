import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model for policy and value network
    """

    def __init__(self):
        super().__init__()

    def soft_update(self, other_network, tau):
        other_variables = other_network.parameters()
        current_variables = self.parameters()

        with torch.no_grad():
            for (current_var, other_var) in zip(current_variables, other_variables):
                current_var.data.copy_(tau * other_var.data + (1.0 - tau) * current_var.data)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)
