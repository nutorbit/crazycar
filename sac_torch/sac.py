import torch
import numpy as np

from torch.optim import Adam
from tqdm import trange

from sac_torch.model import Actor, Critic, ActorCNN, CriticCNN


class SAC:
    """
    Ref: https://arxiv.org/pdf/1812.05905.pdf
    """
    def __init__(self, obs_dim, action_space,
                 gamma=0.99,
                 tau=0.05,
                 lr=3e-4,
                 alpha=0.2,
                 target_update_interval=1,
                 device='cuda'):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval

        self.device = device

        # critic
        self.critic = Critic(obs_dim=obs_dim, act_dim=action_space.shape[0]).to(self.device)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # critic target
        self.critic_target = Critic(obs_dim=obs_dim, act_dim=action_space.shape[0]).to(self.device)
        self.critic_target.hard_update(self.critic)

        # actor
        self.actor = Actor(obs_dim=obs_dim, act_dim=action_space.shape[0], action_space=action_space).to(self.device)
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=lr)

    def select_action(self, obs, evaluate=False):
        obs = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.actor.sample(obs)
        else:
            _, _, action = self.actor.sample(obs)
        return action.detach().cpu().numpy()[0]

    def compute_td_error(self, obs, act, next_obs, rew, done):
        with torch.no_grad():
            next_act, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rew + ((1 - done) * self.gamma * target_q)

        current_q1, current_q2 = self.critic(obs, act)

        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q

        return td_error1, td_error2

    def critic_loss(self, obs, act, next_obs, rew, done):
        td_error1, td_error2 = self.compute_td_error(obs, act, next_obs, rew, done)

        # MSE
        loss1 = (td_error1 ** 2).mean()
        loss2 = (td_error2 ** 2).mean()

        # TODO: use PER instead of Experience replay

        return loss1, loss2

    def actor_alpha_loss(self, obs):

        act, log_prob, _ = self.actor.sample(obs)

        current_q1, current_q2 = self.critic(obs, act)
        min_q = torch.min(current_q1, current_q2)

        actor_loss = ((self.alpha * log_prob) - min_q).mean()

        # alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        return actor_loss, alpha_loss

    def update_critic(self, obs, act, next_obs, rew, done):
        loss1, loss2 = self.critic_loss(obs, act, next_obs, rew, done)

        # update q1
        self.critic_opt.zero_grad()
        loss1.backward(retain_graph=True)
        self.critic_opt.step()

        # update q2
        self.critic_opt.zero_grad()
        loss2.backward(retain_graph=True)
        self.critic_opt.step()

        return loss1, loss2

    def update_actor_alpha(self, obs):
        actor_loss, alpha_loss = self.actor_alpha_loss(obs)

        # update actor
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # update alpha
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        return actor_loss, alpha_loss

    def update_parameters(self, memory, batch_size, updates):
        batch = memory.sample(batch_size)

        # to tensor
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).to(self.device)
        done = torch.FloatTensor(batch['done']).to(self.device)

        # update actor & critic & alpha
        q1_loss, q2_loss = self.update_critic(obs, act, next_obs, rew, done)
        actor_loss, alpha_loss = self.update_actor_alpha(obs)

        # apply alpha
        self.alpha = self.log_alpha.exp()

        # update target network
        if updates % self.target_update_interval == 0:
            self.critic_target.soft_update(self.critic, self.tau)

        return q1_loss, q2_loss, actor_loss, alpha_loss, self.alpha.clone()

    def load_model(self, actor, critic):
        self.actor = actor
        self.critic = critic


def eval(env, agent):
    rews, steps = [], []
    for PosIndex in range(1, 1 + 1):
        obs = env.reset(PosIndex=PosIndex, random_position=False)
        done = False
        episode_reward, episode_steps = 0, 0
        while not done:
            act = agent.select_action(obs, evaluate=True)
            # print(act)
            obs, rew, done, _ = env.step(act)
            episode_reward += rew
            episode_steps += 1
        steps.append(episode_steps)
        rews.append(episode_reward)

    print(f'[EVALUATION] mean_reward: {np.mean(rews)}, mean_steps: {np.mean(steps)}')

    return np.mean(rews), np.mean(steps)


def run(batch_size=256,
        replay_size=int(1e6),
        n_steps=int(2e5),
        start_steps=10000,
        gamma=0.99,
        tau=0.05,
        lr=3e-4,
        alpha=0.2,
        target_update_interval=1,
        steps_per_epochs=4000
        ):

    from pysim.environment import SingleControl, CrazyCar
    from cpprb import ReplayBuffer
    from sac_torch.utils import get_default_rb_dict, Logger

    env = SingleControl(renders=False)
    agent = SAC(
        obs_dim=env.observation_space.shape[0],
        action_space=env.action_space,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha=alpha,
        target_update_interval=target_update_interval
    )

    # define experience replay
    rb_kwargs = get_default_rb_dict(env.observation_space.shape, env.action_space.shape, replay_size)
    rb = ReplayBuffer(**rb_kwargs)

    logger = Logger()

    # save hyperparameter
    logger.save_hyperparameter(
        algorithm='SAC',
        env=env.__class__.__name__,
        batch_size=batch_size,
        replay_size=replay_size,
        n_steps=n_steps,
        start_steps=start_steps,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha=alpha,
        target_update_interval=target_update_interval,
        steps_per_epochs=steps_per_epochs
    )

    logger.start()

    updates = 0
    best_to_save = float('-inf')

    episode_rew, episode_steps = 0, 0
    obs = env.reset(random_position=False)

    for t in trange(n_steps):

        if t < start_steps:
            act = env.action_space.sample()
        else:
            act = agent.select_action(obs)

        next_obs, rew, done, _ = env.step(act)

        episode_rew += rew
        episode_steps += 1

        rb.add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)
        # td_error1, td_error2 = agent.compute_td_error(obs, act, next_obs, rew, done)

        obs = next_obs

        # reset when terminated
        if done:
            obs = env.reset(random_position=False)

            logger.store('Reward/train', episode_rew)
            logger.store('Steps/train', episode_steps)

            episode_rew, episode_steps = 0, 0

        # update nn
        if rb.get_stored_size() > batch_size:
            q1_loss, q2_loss, actor_loss, alpha_loss, alpha = agent.update_parameters(rb,  batch_size, updates)

            logger.store('Loss/Q1', q1_loss)
            logger.store('Loss/Q2', q2_loss)
            logger.store('Loss/Actor', actor_loss)
            logger.store('Loss/Alpha', alpha_loss)
            logger.store('Param/Alpha', alpha)

            updates += 1

        # eval and save
        if (t+1) % steps_per_epochs == 0:

            # test
            mean_rew, mean_steps = eval(env, agent)
            logger.store('Reward/test', mean_rew)
            logger.store('Steps/test', mean_steps)

            # save a model
            if best_to_save <= mean_steps:
                best_to_save = mean_steps
                logger.save_model([agent.actor, agent.critic])

        logger.update_steps()


if __name__ == '__main__':
    run()


