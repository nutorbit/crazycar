import torch
import torch.nn as nn

from copy import deepcopy
from torch.optim import Adam

from crazycar.utils import make_mlp, weight_init
from crazycar.algos.model import BaseModel


class Actor(BaseModel):
    """
    Actor for DDPG

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = deepcopy(encoder)
        self.pi = make_mlp(sizes=[self.enc.out_size] + hiddens + [act_dim], activation=nn.ReLU)
        self.apply(weight_init)

    def forward(self, obs):
        x = self.enc(obs)
        x = self.pi(x)
        return x


class Critic(BaseModel):
    """
    Double Q for DDPG

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = deepcopy(encoder)
        self.q1 = make_mlp(sizes=[self.enc.out_size + act_dim] + hiddens + [1], activation=nn.ReLU)
        self.q2 = make_mlp(sizes=[self.enc.out_size + act_dim] + hiddens + [1], activation=nn.ReLU)
        self.apply(weight_init)

    def forward(self, obs, act):
        x = self.enc(obs)
        x = torch.cat([x, act], dim=1)
        return self.q1(x), self.q2(x)


class DDPG:
    """
    Deep Deterministic Policy Gradient

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        lr: learning rate
        gamma: discount factor
        interval_target: number of iteration for update target network
        tau: polyak average
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder,
                 act_dim,
                 lr=1e-4,
                 gamma=0.9,
                 interval_target=2,
                 tau=0.05,
                 hiddens=[256, 256]):

        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        # define actor
        self.actor = Actor(encoder, act_dim, hiddens)
        self.actor_target = deepcopy(self.actor)
        self.actor_target.hard_update(self.actor)

        # define critic
        self.critic = Critic(encoder, act_dim, hiddens)
        self.critic_target = deepcopy(self.critic)
        self.critic_target.hard_update(self.critic)

        # define optimizer
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

    def actor_loss(self, batch):
        """
        L(s) = -E[Q(s, a)| a~u(s)]
        """

        act = self.actor(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        loss = -q1
        return loss

    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2

        Where,
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s')
        """

        with torch.no_grad():
            next_act = self.actor_target(batch['next_obs'])
            q_target1, q_target2 = self.critic_target(batch['next_obs'], next_act)
            q_target = torch.min(q_target1, q_target2)
            y = batch['rew'] + (1 - batch['done']) * self.gamma * q_target

        q1, q2 = self.critic(batch['obs'], batch['act'])

        loss1 = (y - q1).pow(2).mean()
        loss2 = (y - q2).pow(2).mean()

        return loss1 + loss2

    def update_actor(self, batch):
        loss = self.actor_loss(batch)

        # backward
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        return loss

    def update_critic(self, batch):
        loss = self.critic_loss(batch)

        # backward
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        return loss

    def update_params(self, batch, i):
        critic_loss = self.update_actor(batch)
        actor_loss = self.update_actor(batch)

        # update target network
        if i % self.interval_target == 0:
            self.actor_target.soft_update(self.actor, self.tau)
            self.critic_target.soft_update(self.critic, self.tau)

        return {
            "actor_loss": actor_loss.detach(),
            "critic_loss": critic_loss.detach()
        }

    def predict(self, obs):
        act = self.actor(obs)
        return act
