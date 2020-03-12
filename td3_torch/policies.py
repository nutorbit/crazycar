import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from td3_torch.utils import set_seed, make_mlp


# set_seed(100)


class BaseModel(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def soft_update(self, other_network, polyak):

        other_variables = other_network.parameters()
        current_variables = self.parameters()

        with torch.no_grad():
            for (current_var, other_var) in zip(current_variables, other_variables):
                current_var.data.mul_(polyak)
                current_var.data.add_((1 - polyak) * other_var.data)

    def hard_update(self, other_network):
        self.soft_update(other_network, polyak=1.)


class Actor(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        sizes = [obs_dim] + [256, 256, 256] + [act_dim]
        self.pi = make_mlp(sizes=sizes, activation=nn.ReLU, output_activation=nn.Tanh).cuda()

    def forward(self, obs):
        return self.pi(obs)


class ActorCNN(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.cnn = ImpalaCNN(obs_dim).cuda()
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            # nn.Linear(128, 64), nn.ReLU(),
            # nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh()
        ).cuda()

    def forward(self, x):
        x = self.cnn(x/255)
        # print(x)
        x = self.fc(x)
        # print(self.fc.parameters())
        # print(x)
        return x


class Critic(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        sizes = [obs_dim + act_dim] + [256, 256, 256] + [1]
        self.q1 = make_mlp(sizes=sizes, activation=nn.ReLU).cuda()
        self.q2 = make_mlp(sizes=sizes, activation=nn.ReLU).cuda()

    def forward(self, obs, act):
        concat = torch.cat([obs, act], dim=1)
        return [self.q1(concat), self.q2(concat)]

    def q1_forward(self, obs, act):
        concat = torch.cat([obs, act], dim=1)
        return self.q1(concat)


class DuelingQ(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.body = make_mlp([obs_dim] + [256, 128], nn.ReLU, nn.ReLU).cuda()
        self.v = make_mlp([128, 1], nn.Identity).cuda()
        self.a = make_mlp([obs_dim + act_dim, 128, 1], nn.Identity).cuda()

    def forward(self, obs, act):

        feature = self.body(obs)

        concat = torch.cat([obs, act], dim=1)
        v = self.v(feature)
        a = self.a(concat)

        out = v + a

        return out


class DuelingCritic(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.q1 = DuelingQ(obs_dim, act_dim).cuda()
        self.q2 = DuelingQ(obs_dim, act_dim).cuda()

    def forward(self, obs, act):
        return [self.q1(obs, act), self.q2(obs, act)]

    def q1_forward(self, obs, act):
        return self.q1(obs, act)


class CriticCNN(BaseModel):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.cnn = ImpalaCNN(obs_dim).cuda()
        # self.cnn_q1 = ImpalaCNN(image_size)
        self.q1 = nn.Sequential(
            nn.Linear(256 + act_dim, 64), nn.ReLU(),
            # nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(64, 1)
        ).cuda()

        # self.cnn_q2 = ImpalaCNN(image_size)
        self.q2 = nn.Sequential(
            nn.Linear(256 + act_dim, 64), nn.ReLU(),
            # nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(64, 1)
        ).cuda()

    def forward(self, obs, act):
        # cnn_out_q1 = self.cnn_q1(obs)
        # cnn_out_q2 = self.cnn_q2(obs)

        # concat_1 = torch.cat([cnn_out_q1, act], dim=1)
        # concat_2 = torch.cat([cnn_out_q2, act], dim=1)

        # concat = torch.cat([obs, act], dim=1)
        # return [self.q1(concat_1), self.q2(concat_2)]

        cnn_out = self.cnn(obs/255)

        concat = torch.cat([cnn_out, act], dim=1)
        return [self.q1(concat), self.q2(concat)]

    def q1_forward(self, obs, act):

        cnn_out = self.cnn(obs/255)

        concat = torch.cat([cnn_out, act], dim=1)

        return self.q1(concat)


class ActorCritic:
    def __init__(self, obs_dim, act_dim, actor_lr=1e-4, critic_lr=1e-4):
        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.hard_update(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.hard_update(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs)


class ActorCriticCNN:
    def __init__(self, obs_dim, act_dim, actor_lr=1e-4, critic_lr=1e-4):
        self.actor = ActorCNN(obs_dim, act_dim)
        self.actor_target = ActorCNN(obs_dim, act_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = CriticCNN(obs_dim, act_dim)
        self.critic_target = CriticCNN(obs_dim, act_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs)


class ImpalaCNN(nn.Module):
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
        # print(x.shape)
        x = self.conv_layers(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = F.relu(x)
        # print(x)
        return x


class ImpalaResidual(nn.Module):
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


class FixupCNN(nn.Module):
    def __init__(self, image_size, depth_in=4):
        super().__init__()
        layers = []
        for depth_out in [32, 64, 64]:
            layers.extend([
                nn.Conv2d(depth_in, depth_out, 3, padding=1),
                nn.MaxPool2d(3, stride=2, padding=1),
                FixupResidual(depth_out, 8),
                FixupResidual(depth_out, 8),
            ])
            depth_in = depth_out
        layers.extend([
            FixupResidual(depth_in, 8),
            FixupResidual(depth_in, 8),
        ])
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


class FixupResidual(nn.Module):
    def __init__(self, depth, num_residual):
        super().__init__()
        self.conv1 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(depth, depth, 3, padding=1, bias=False)
        for p in self.conv1.parameters():
            p.data.mul_(1 / math.sqrt(num_residual))
        for p in self.conv2.parameters():
            p.data.zero_()
        self.bias1 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias2 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias3 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.bias4 = nn.Parameter(torch.zeros([depth, 1, 1]))
        self.scale = nn.Parameter(torch.ones([depth, 1, 1]))

    def forward(self, x):
        x = F.relu(x)
        out = x + self.bias1
        out = self.conv1(out)
        out = out + self.bias2
        out = F.relu(out)
        out = out + self.bias3
        out = self.conv2(out)
        out = out * self.scale
        out = out + self.bias4
        return out + x


if __name__ == "__main__":
    # a = ActorCritic(3, 1)
    # obs = torch.as_tensor(np.array([1, 2, 3]).reshape((1, 3)), dtype=torch.float32)
    # act = torch.as_tensor(np.array([0.1]).reshape((1, 1)), dtype=torch.float32)
    # print(a.c(obs, act))
    from pysim.environment import CrazyCar, SingleControl

    env = SingleControl()

    obs = env.reset()
    print(obs.shape)

    ac = ActorCriticCNN(28, 1)

    obs = torch.as_tensor(obs, dtype=torch.float64).unsqueeze(0)

    print(obs.shape)
    # print(obs.unsqueeze(0).shape)
    print(ac.act(obs.cuda()))
