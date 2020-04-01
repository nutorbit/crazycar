import torch
import numpy as np
import torch.nn.functional as F

from torch.optim import Adam

from sac_torch.model import Actor, Critic


class SAC:
    def __init__(self, obs_dim, action_space,
                 gamma=0.99,
                 tau=0.05,
                 lr=3e-4,
                 alpha=0.2,
                 batch_size=256,
                 target_update_interval=1,
                 device='cuda'):

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval
        self.batch_size = batch_size

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

        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2

        return td_error1, td_error2

    def critic_loss(self, obs, act, next_obs, rew, done):
        td_error1, td_error2 = self.compute_td_error(obs, act, next_obs, rew, done)

        # MSE
        loss1 = (td_error1 ** 2).mean()
        loss2 = (td_error2 ** 2).mean()

        # TODO: use PER instead of Experience replay

        return loss1, loss2

    def actor_loss(self, obs):

        act, log_prob, _ = self.actor.sample(obs)

        current_q1, current_q2 = self.critic(obs, act)
        min_q = torch.min(current_q1, current_q2)

        loss = (min_q - (self.alpha * log_prob)).mean()

        return -loss

    def update_critic(self, obs, act, next_obs, rew, done):
        loss1, loss2 = self.critic_loss(obs, act, next_obs, rew, done)

        # update q1
        self.critic_opt.zero_grad()
        loss1.backward()
        self.critic_opt.step()

        # update q2
        self.critic_opt.zero_grad()
        loss2.backward()
        self.critic_opt.step()

        return loss1, loss2

    def update_actor(self, obs):
        loss = self.actor_loss(obs)

        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()

        return loss

    def update_parameters(self, memory, batch_size, updates):
        batch = memory.sample(batch_size)

        # to tensor
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).to(self.device)
        done = torch.FloatTensor(batch['done']).to(self.device)

        # update actor & critc
        q1_loss, q2_loss = self.update_critic(obs, act, next_obs, rew, done)
        actor_loss = self.update_actor(obs)

        # update target network
        if updates % self.target_update_interval == 0:
            self.critic.soft_update(self.critic, self.tau)

        return q1_loss, q2_loss, actor_loss


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


def main():

    from pysim.environment import SingleControl
    from cpprb import ReplayBuffer
    from sac_torch.utils import get_default_rb_dict, Logger

    env = SingleControl(renders=False)
    agent = SAC(env.observation_space.shape[0], env.action_space)

    # define experience replay
    rb_kwargs = get_default_rb_dict(env.observation_space.shape, env.action_space.shape, 100000)
    rb = ReplayBuffer(**rb_kwargs)

    logger = Logger()
    logger.start()

    total_numsteps = 0
    updates = 0

    for i_episode in range(1000):
        episode_rew = 0
        episode_steps = 0
        done = False
        obs = env.reset(random_position=False)

        while not done:
            if total_numsteps < 10000:
                act = env.action_space.sample()
            else:
                act = agent.select_action()
            # print(act)
            if rb.get_stored_size() > 256:
                q1_loss, q2_loss, actor_loss = agent.update_parameters(rb,  256, updates)

                logger.store('Loss/Q1', q1_loss)
                logger.store('Loss/Q2', q2_loss)
                logger.store('Loss/Actor', actor_loss)

                updates += 1

            next_obs, rew, done, _ = env.step(act)
            episode_steps += 1
            total_numsteps += 1
            episode_rew += 1

            rb.add(obs=obs, next_obs=next_obs, act=act, rew=rew, done=done)

            obs = next_obs

            logger.update_steps()

        logger.store('Reward/train', episode_rew)
        logger.store('Steps/train', episode_steps)

        # test
        mean_rew, mean_steps = eval(env, agent)
        logger.store('Reward/test', mean_rew)
        logger.store('Steps/test', mean_steps)

        # TODO: add save a model


if __name__ == '__main__':
    main()


