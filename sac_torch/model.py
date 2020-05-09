import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from sac_torch.utils import make_mlp, weight_init


class BaseModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def soft_update(self, other_network, tau):

        other_variables = other_network.parameters()
        current_variables = self.parameters()

        with torch.no_grad():
            for (current_var, other_var) in zip(current_variables, other_variables):
                current_var.data.copy_(tau * other_var.data + (1.0 - tau) * current_var.data)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)


class Critic(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        sizes = [obs_dim + act_dim] + [256, 256] + [1]
        self.q1 = make_mlp(sizes=sizes, activation=nn.ReLU)
        self.q2 = make_mlp(sizes=sizes, activation=nn.ReLU)

        self.apply(weight_init)

    def forward(self, obs, act):
        concat = torch.cat([obs, act], dim=1)
        return self.q1(concat), self.q2(concat)


class Actor(BaseModel):
    def __init__(self, obs_dim, act_dim, action_space=None):
        super().__init__(obs_dim, act_dim)
        sizes = [obs_dim] + [256, 256]
        self.hidden = make_mlp(sizes, activation=nn.ReLU, output_activation=nn.ReLU)
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

        self.apply(weight_init)

    def forward(self, obs):
        x = self.hidden(obs)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class ActorCNN(Actor):
    def __init__(self, obs_dim, act_dim, action_space=None):
        super().__init__(obs_dim, act_dim, action_space)
        self.cnn = ImpalaCNN(obs_dim)
        sizes = [256, 256, 256]
        self.hidden = make_mlp(sizes, activation=nn.ReLU, output_activation=nn.ReLU)

        self.apply(weight_init)

    def forward(self, obs):
        x = self.cnn(obs/255)
        x = self.hidden(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std


class CriticCNN(Critic):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.cnn = ImpalaCNN(obs_dim)
        sizes = [256 + act_dim] + [256, 256, 256] + [1]
        self.q1 = make_mlp(sizes=sizes, activation=nn.ReLU)
        self.q2 = make_mlp(sizes=sizes, activation=nn.ReLU)

        self.apply(weight_init)

    def forward(self, obs, act):
        x = self.cnn(obs/255).view((obs.shape[0], -1))  # flatten
        concat = torch.cat([x, act], dim=1)

        return self.q1(concat), self.q2(concat)


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
