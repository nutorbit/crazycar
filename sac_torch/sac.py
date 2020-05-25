import torch
import logging
import numpy as np

from datetime import datetime
from torch.optim import Adam
from tqdm import trange
from cpprb import ReplayBuffer

from sac_torch.model import Actor, Critic, ActorCNN, CriticCNN
from sac_torch.utils import set_seed_everywhere, huber_loss, get_helper_logger, get_default_rb_dict, Logger
from pysim.environment import SingleControl, CrazyCar, FrameStack


class SAC:
    """
    Ref: https://arxiv.org/pdf/1812.05905.pdf
    """
    def __init__(self, observation_space, action_space,
                 date=None,
                 replay_size=int(1e6),
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
        self.logger = None

        # experience replay
        rb_kwargs = get_default_rb_dict(observation_space.shape, action_space.shape, replay_size)
        self.rb = ReplayBuffer(**rb_kwargs)

        # logger
        if date is not None:
            self.logger = get_helper_logger('SAC', date)
            self.logger.info("SAC algorithm has started")
            self.logger.info(f"gamma: {str(gamma)}")
            self.logger.info(f"tau: {str(tau)}")
            self.logger.info(f"alpha: {str(alpha)}")
            self.logger.info(f"target_update_interval: {str(target_update_interval)}")
            self.logger.info(f"device: {str(device)}")

        # critic
        self.critic = CriticCNN(obs_dim=observation_space.shape[0], act_dim=action_space.shape[0]).to(self.device)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # critic target
        self.critic_target = CriticCNN(obs_dim=observation_space.shape[0], act_dim=action_space.shape[0]).to(self.device)
        self.critic_target.hard_update(self.critic)

        # actor
        self.actor = ActorCNN(obs_dim=observation_space.shape[0], act_dim=action_space.shape[0], action_space=action_space).to(self.device)
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
        loss1 = huber_loss(td_error1).mean()
        loss2 = huber_loss(td_error2).mean()

        if self.logger is not None:
            self.logger.debug(f'Critic loss1: {loss1}')
            self.logger.debug(f'Critic loss2: {loss2}')

        return loss1 + loss2

    def actor_alpha_loss(self, obs):

        act, log_prob, _ = self.actor.sample(obs)

        current_q1, current_q2 = self.critic(obs, act)
        min_q = torch.min(current_q1, current_q2)

        actor_loss = ((self.alpha * log_prob) - min_q).mean()

        # alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        if self.logger is not None:
            self.logger.debug(f'Actor loss: {actor_loss}')
            self.logger.debug(f'Alpha loss: {alpha_loss}')

        return actor_loss, alpha_loss

    def update_critic(self, obs, act, next_obs, rew, done):
        critic_loss = self.critic_loss(obs, act, next_obs, rew, done)

        # update critic
        self.critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()

        return critic_loss

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

    def update_parameters(self, batch_size, updates):
        batch = self.rb.sample(batch_size)

        # to tensor
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).to(self.device)
        done = torch.FloatTensor(batch['done']).to(self.device)

        # update actor & critic & alpha
        critic_loss = self.update_critic(obs, act, next_obs, rew, done)
        actor_loss, alpha_loss = self.update_actor_alpha(obs)

        # apply alpha
        self.alpha = self.log_alpha.exp()

        # update target network
        if updates % self.target_update_interval == 0:
            self.critic_target.soft_update(self.critic, self.tau)

        return critic_loss, actor_loss, alpha_loss, self.alpha.clone()

    def load_model(self, actor, critic):
        self.actor = actor
        self.critic = critic

        if self.logger is not None:
            self.logger.info('Weight loading')


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

    return np.mean(rews), np.mean(steps)


def run(batch_size=256,
        replay_size=int(1e6),
        n_steps=int(2e5),
        start_steps=10000,
        gamma=0.98,
        tau=0.05,
        lr=3e-4,
        alpha=0.2,
        target_update_interval=2,
        steps_per_epochs=4000,
        seed=100
        ):

    set_seed_everywhere(seed)
    date = datetime.now().strftime("%b_%d_%Y_%H%M%S")
    logger_main = get_helper_logger('Main', date)
    logger_main.info(f'Process has started')
    logger_main.info(f'batch_size: {str(batch_size)}')
    logger_main.info(f'replay_size: {str(replay_size)}')
    logger_main.info(f'n_steps: {str(n_steps)}')
    logger_main.info(f'gamma: {str(gamma)}')
    logger_main.info(f'tau: {str(tau)}')
    logger_main.info(f'lr: {str(lr)}')
    logger_main.info(f'alpha: {str(alpha)}')
    logger_main.info(f'target_update_interval: {str(target_update_interval)}')
    logger_main.info(f'steps_per_epochs: {str(steps_per_epochs)}')
    logger_main.info(f'seed: {str(seed)}')

    env = SingleControl(renders=True, date=date, track_id=2)
    env = FrameStack(env)
    logger_main.info(f'Environment: {str(env.__class__.__name__)}')
    logger_main.info(f"-----------------")

    agent = SAC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        date=date,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha=alpha,
        target_update_interval=target_update_interval
    )

    logger = Logger(date)

    # save hyperparameter
    logger.save_hyperparameter(
        algorithm='SAC',
        agent=agent.actor.__class__.__name__,
        env=env.__class__.__name__,
        reward=env._reward_function,
        batch_size=batch_size,
        replay_size=replay_size,
        n_steps=n_steps,
        start_steps=start_steps,
        gamma=gamma,
        tau=tau,
        lr=lr,
        alpha=alpha,
        target_update_interval=target_update_interval,
        steps_per_epochs=steps_per_epochs,
        seed=seed
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

        agent.rb.add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)

        obs = next_obs

        # reset when terminated
        if done:
            n_collision = env.report()

            obs = env.reset(random_position=False)

            logger.store('Reward/train', episode_rew)
            logger.store('Steps/train', episode_steps)
            logger.store('N_Collision/train', n_collision)
            logger_main.info(f"End of episode at {t}")
            logger_main.info(f"Reward: {episode_rew}")
            logger_main.info(f"Step:s {episode_steps}")
            logger_main.info(f"-----------------")

            episode_rew, episode_steps = 0, 0

        # update nn
        if agent.rb.get_stored_size() > batch_size:
            critic_loss, actor_loss, alpha_loss, alpha = agent.update_parameters(batch_size, updates)

            logger.store('Loss/Critic', critic_loss)
            logger.store('Loss/Actor', actor_loss)
            logger.store('Loss/Alpha', alpha_loss)
            logger.store('Param/Alpha', alpha)

            updates += 1

        # eval and save
        if (t+1) % steps_per_epochs == 0:
            n_collision = env.report()

            # test
            mean_rew, mean_steps = eval(env, agent)
            logger.store('Reward/test', mean_rew)
            logger.store('Steps/test', mean_steps)
            logger.store('N_Collision/test', n_collision)
            logger_main.info(f"Evaluation at {t}")
            logger_main.info(f"Reward: {str(mean_rew)}")
            logger_main.info(f"Steps: {str(mean_steps)}")
            logger_main.info(f"N_Collision: {str(n_collision)}")
            logger_main.info(f"-----------------")

            # save a model
            if best_to_save <= mean_rew:
                best_to_save = mean_rew
                logger.save_model([agent.actor, agent.critic])

        logger.update_steps()


if __name__ == '__main__':
    run()


