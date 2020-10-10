import torch
import torch.nn as nn

from torch.optim import Adam
from torch.distributions import Normal

from crazycar.algos_torch.base import BaseNetwork, BaseModel
from crazycar.algos_torch.common import make_mlp


class Actor(BaseNetwork):
    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = encoder()
        self.act_dim = act_dim
        self.hidden = make_mlp(
            sizes=[256] + hiddens,
            activation=nn.ReLU
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, obs):
        x = self.enc(obs)
        x = self.hidden(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
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


class Critic(BaseNetwork):
    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = encoder()
        self.act_dim = act_dim
        self.q1 = make_mlp(
            sizes=[256 + act_dim] + hiddens,
            activation=nn.ReLU
        )
        self.q2 = make_mlp(
            sizes=[256 + act_dim] + hiddens,
            activation=nn.ReLU
        )

    def forward(self, obs, act):
        x = self.enc(obs)
        concat = torch.cat([x, act], dim=1)
        return self.q1(concat), self.q2(concat)


class CriticV(BaseNetwork):
    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = encoder()
        self.act_dim = act_dim
        self.q1 = make_mlp(
            sizes=[256] + hiddens,
            activation=nn.ReLU
        )
        self.q2 = make_mlp(
            sizes=[256] + hiddens,
            activation=nn.ReLU
        )

    def forward(self, obs):
        x = self.enc(obs)
        return self.q1(x), self.q2(x)


class SAC(BaseModel):

    def __init__(self, encoder, act_dim,
                 lr=3e-4,
                 gamma=0.99,
                 interval_target=2,
                 tau=0.05,
                 replay_size=int(1e6),
                 hiddens=[256, 256]):

        super().__init__(replay_size=replay_size)

        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        # define actor
        self.actor = Actor(encoder, act_dim, hiddens).to('cuda')
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)

        # define critic
        self.critic = Critic(encoder, act_dim, hiddens).to('cuda')
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # define critic v
        self.critic_v = CriticV(encoder, act_dim, hiddens).to('cuda')
        self.critic_v_target = CriticV(encoder, act_dim, hiddens).to('cuda')
        self.critic_v_target.hard_update(self.critic_v)
        self.critic_v_opt = Adam(self.critic_v.parameters(), lr=lr)

        # define alpha
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.target_entropy = -torch.Tensor(act_dim).to('cuda')
        self.alpha_opt = Adam([self.log_alpha], lr=lr)

    def actor_loss(self, batch):
        """
        L(s) = -E[Q(s, a)| a~u(s)]

        Where,
            Q is a soft-Q: Q - alpha * log_prob
        """

        act, log_prob, _ = self.actor.sample(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        min_q = torch.min(q1, q2)
        loss = (torch.exp(self.log_alpha) * log_prob - min_q).mean()

        return loss

    def critic_v_loss(self, batch):
        """
        ...
        """

        v1, v2 = self.critic_v(batch['obs'])
        v = torch.min(v1, v2)

        act, log_prob, _ = self.actor.sample(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        min_q = torch.min(q1, q2)

        with torch.no_grad():
            target_v = min_q - torch.exp(self.log_alpha) * log_prob
        td_v = ((target_v - v) ** 2).mean()

        return td_v

    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2

        Where,
            Q is a soft-Q: Q - alpha * log_prob
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s')
        """

        q1, q2 = self.critic(batch['obs'], batch['act'])
        next_v_target1, next_v_target2 = self.critic_v(batch['next_obs'])
        next_v_target = torch.min(next_v_target1, next_v_target2)

        with torch.no_grad():
            target_q = batch['rew'] + (1 - batch['done']) * self.gamma * next_v_target

        td_q1 = ((target_q - q1) ** 2).mean()
        td_q2 = ((target_q - q2) ** 2).mean()

        return td_q1 + td_q2

    def alpha_loss(self, batch):
        """
        L = -(alpha * log_prob + target_entropy)
        """

        act, log_prob, _ = self.actor.sample(batch['obs'])
        # print(act, log_prob)
        loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        return loss

    def update_params(self, i, batch_size=256):
        batch = self.rb.sample(batch_size)

        critic_loss = self._update(self.critic_opt, self.critic_loss, batch)
        critic_v_loss = self._update(self.critic_v_opt, self.critic_v_loss, batch)
        actor_loss = self._update(self.actor_opt, self.actor_loss, batch)
        alpha_loss = self._update(self.alpha_opt, self.alpha_loss, batch)

        # update target network
        if i % self.interval_target == 0:
            self.critic_v_target.soft_update(self.critic_v, self.tau)

        return {
            "actor_loss": actor_loss.cpu().detach().numpy(),
            "critic_loss": critic_loss.cpu().detach().numpy(),
            "critic_v_loss": critic_v_loss.cpu().detach().numpy(),
            "alpha_loss": alpha_loss.cpu().detach().numpy(),
            "alpha": torch.exp(self.log_alpha).cpu().detach().numpy()
        }

    def predict(self, obs):
        act, _, _ = self.actor.sample(obs)
        act = act.cpu().detach().numpy()
        if len(act[0]) == 2:
            # apply rescale to speed
            act[0][0] = self.rescale(act[0][0])
        return act

    def write_metric(self, writer, metric, step):
        writer.add_scalar("loss/actor_loss", metric['actor_loss'], step)
        writer.add_scalar("loss/critic_loss", metric['critic_loss'], step)
        writer.add_scalar("loss/critic_v_loss", metric['critic_v_loss'], step)
        writer.add_scalar("loss/alpha_loss", metric['alpha_loss'], step)
        writer.add_scalar("track/alpha", metric['alpha'], step)



