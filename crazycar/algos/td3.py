import torch

from crazycar.algos.ddpg import DDPG


class TD3(DDPG):
    """
    Twin Delayed Deep Deterministic Policy Gradient

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        lr: learning rate
        gamma: discount factor
        interval_target: number of iteration for update target network
        tau: polyak average
        hiddens: NO. units for each layers
        target_noise: noise in target network
        noise_clip: noise clip
    """

    def __init__(self, encoder,
                 act_dim,
                 lr=1e-4,
                 gamma=0.9,
                 interval_target=2,
                 tau=0.05,
                 hiddens=[256, 256],
                 target_noise=0.1,
                 noise_clip=0.1):

        super().__init__(encoder, act_dim, lr, gamma, interval_target, tau, hiddens)
        self.target_noise = target_noise
        self.noise_clip = noise_clip

    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2

        Where,
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s') + noise
        """

        with torch.no_grad():
            # next action
            next_act = self.actor_target(batch['next_obs'])
            noise = torch.randn_like(next_act) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_act = torch.clamp(next_act + noise, -1, 1)

            q_target1, q_target2 = self.critic_target(batch['next_obs'], next_act)
            q_target = torch.min(q_target1, q_target2)
            y = batch['rew'] + (1 - batch['done']) * self.gamma * q_target

        q1, q2 = self.critic(batch['obs'], batch['act'])

        loss1 = (y - q1).pow(2).mean()
        loss2 = (y - q2).pow(2).mean()

        return loss1 + loss2
