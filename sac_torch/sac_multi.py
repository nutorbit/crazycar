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
        self.critic = CriticCNN(obs_dim=obs_dim, act_dim=action_space.shape[0]).to(self.device)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # critic target
        self.critic_target = CriticCNN(obs_dim=obs_dim, act_dim=action_space.shape[0]).to(self.device)
        self.critic_target.hard_update(self.critic)

        # actor
        self.actor = ActorCNN(obs_dim=obs_dim, act_dim=action_space.shape[0], action_space=action_space).to(self.device)
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


def eval(env, agent1, agent2):
    rews1, rews2, steps = [], [], []
    obs = env.reset()
    done = False
    episode_rew1 = 0
    episode_rew2 = 0
    episode_steps = 0
    while not done:
        act1 = agent1.select_action(obs[0], evaluate=True)
        act2 = agent2.select_action(obs[1], evaluate=True)
        # print(act)
        obs, rew, done, _ = env.step([act1, act2])
        episode_rew1 += rew[0]
        episode_rew2 += rew[1]
        episode_steps += 1
    steps.append(episode_steps)
    rews1.append(episode_rew1)
    rews2.append(episode_rew2)

    print(f'\n[EVALUATION] agent1: {np.mean(rews1)}| agent2 {np.mean(rews2)}, mean_steps: {np.mean(steps)}')

    return np.mean(rews1), np.mean(rews2), np.mean(steps)


def run(batch_size=256,
        replay_size=int(1e6),
        n_steps=int(2e5),
        start_steps=10000,
        gamma=0.98,
        tau=0.05,
        lr=3e-4,
        alpha=0.2,
        target_update_interval=2,
        steps_per_epochs=4000
        ):

    from time import sleep
    from pysim.environment import SingleControl, CrazyCar, MultiCar
    from cpprb import ReplayBuffer
    from sac_torch.utils import get_default_rb_dict, Logger

    env = MultiCar(renders=False)

    agent1 = SAC(
        obs_dim=20,
        action_space=env.action_space,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha=alpha,
        target_update_interval=target_update_interval
    )

    agent2 = SAC(
        obs_dim=20,
        action_space=env.action_space,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha=alpha,
        target_update_interval=target_update_interval
    )

    print(f'Obsevation space: {env.observation_space.shape}')
    print(f'Action space: {env.action_space.shape}')

    # define experience replay
    rb_kwargs = get_default_rb_dict((20, 20, 1), env.action_space.shape, replay_size)
    rb1 = ReplayBuffer(**rb_kwargs)
    rb2 = ReplayBuffer(**rb_kwargs)

    logger1 = Logger()
    sleep(2)
    logger2 = Logger()

    # save hyperparameter
    logger1.save_hyperparameter(
        algorithm='SAC',
        agent=agent1.actor.__class__.__name__,
        shape=env.action_space.shape,
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

    logger2.save_hyperparameter(
        algorithm='SAC',
        agent=agent2.actor.__class__.__name__,
        shape=env.action_space.shape,
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

    logger1.start()
    logger2.start()

    updates = 0
    best_to_save = float('-inf')

    episode_rew1 = 0
    episode_rew2 = 0
    episode_steps = 0
    win = [0, 0]
    obs = env.reset()

    for t in trange(n_steps):

        if t < start_steps:
            act1 = env.action_space.sample()
            act2 = env.action_space.sample()
        else:
            act1 = agent1.select_action(obs[0])
            act2 = agent2.select_action(obs[1])

        next_obs, rew, done, _ = env.step([act1, act2])

        episode_rew1 += rew[0]
        episode_rew2 += rew[1]
        episode_steps += 1
        # print(dict(obs=obs[0].shape, act=act1.shape, next_obs=next_obs[0].shape, rew=rew[0], done=done))
        rb1.add(obs=obs[0], act=act1, next_obs=next_obs[0], rew=rew[0], done=done)
        rb2.add(obs=obs[1], act=act2, next_obs=next_obs[1], rew=rew[1], done=done)
        # td_error1, td_error2 = agent.compute_td_error(obs, act, next_obs, rew, done)

        obs = next_obs

        # reset when terminated
        if done:
            n_collision, win_idx = env.report()

            if win_idx is not None:
                win[win_idx] += 1
                logger1.store('Win_prob', win[0] / sum(win))
                logger2.store('Win_prob', win[1] / sum(win))

            obs = env.reset()

            logger1.store('Reward/train', episode_rew1)
            logger1.store('Steps/train', episode_steps)
            logger1.store('N_Collision/train', n_collision[0])

            logger2.store('Reward/train', episode_rew2)
            logger2.store('Steps/train', episode_steps)
            logger2.store('N_Collision/train', n_collision[1])

            episode_rew1 = 0
            episode_rew2 = 0
            episode_steps = 0

        # update nn
        if rb1.get_stored_size() > batch_size:
            q1_loss1, q2_loss1, actor_loss1, alpha_loss1, alpha1 = agent1.update_parameters(rb1, batch_size, updates)
            q1_loss2, q2_loss2, actor_loss2, alpha_loss2, alpha2 = agent2.update_parameters(rb2, batch_size, updates)

            logger1.store('Loss/Q1', q1_loss1)
            logger1.store('Loss/Q2', q2_loss1)
            logger1.store('Loss/Actor', actor_loss1)
            logger1.store('Loss/Alpha', alpha_loss1)
            logger1.store('Param/Alpha', alpha1)

            logger2.store('Loss/Q1', q1_loss2)
            logger2.store('Loss/Q2', q2_loss2)
            logger2.store('Loss/Actor', actor_loss2)
            logger2.store('Loss/Alpha', alpha_loss2)
            logger2.store('Param/Alpha', alpha2)

            updates += 1

        # eval and save
        if (t+1) % steps_per_epochs == 0:

            # test
            mean_rew1, mean_rew2, mean_steps = eval(env, agent1, agent2)

            n_collision, win_idx = env.report()

            if win_idx is not None:
                win[win_idx] += 1
                logger1.store('Win_prob', win[0]/sum(win))
                logger2.store('Win_prob', win[1]/sum(win))

            logger1.store('Reward/test', mean_rew1)
            logger1.store('Steps/test', mean_steps)
            logger1.store('N_Collision/test', n_collision[0])

            logger2.store('Reward/test', mean_rew2)
            logger2.store('Steps/test', mean_steps)
            logger2.store('N_Collision/test', n_collision[1])

            # save a model
            if best_to_save <= max(mean_rew1, mean_rew2):
                best_to_save = max(mean_rew1, mean_rew2)
                logger1.save_model([agent1.actor, agent1.critic])
                logger2.save_model([agent2.actor, agent2.critic])

        logger1.update_steps()
        logger2.update_steps()


if __name__ == '__main__':
    run()


