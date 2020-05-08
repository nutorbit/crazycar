import torch
import numpy as np
import notify

from datetime import datetime
from tqdm import trange

from td3_torch.utils import Logger, huber_loss, get_default_rb_dict, get_helper_logger, set_seed_everywhere
from td3_torch.policies import ActorCritic, ActorCriticCNN

from pysim.environment import CrazyCar, SingleControl, FrameStack

from cpprb import ReplayBuffer


class TD3:
    def __init__(self, observation_space, action_space,
                 date=None,
                 tau=0.005,
                 gamma=0.9,
                 target_noise=0.2,
                 replay_size=100000,
                 noise_clip=0.5,
                 policy_delay=2,
                 name='Anonymous',
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 device='cuda'
                 ):

        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space
        self.last_actor_loss = 0

        self.tau = tau
        self.gamma = gamma
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device

        # logger
        if date is not None:
            self.logger = get_helper_logger('TD3', date)
            self.logger.info("TD3 algorithm has started")
            self.logger.info(f"tau: {str(tau)}")
            self.logger.info(f"gamma: {str(gamma)}")
            self.logger.info(f"target_noise: {str(target_noise)}")
            self.logger.info(f"replay_size: {str(replay_size)}")
            self.logger.info(f"noise_clip: {str(noise_clip)}")
            self.logger.info(f"policy_delay: {str(policy_delay)}")
            self.logger.info(f"actor_lr: {str(actor_lr)}")
            self.logger.info(f"critic_lr: {str(critic_lr)}")
            self.logger.info(f"device: {str(device)}")

        self.ac = ActorCriticCNN(observation_space.shape[0], action_space.shape[0], actor_lr, critic_lr, device=device)

        rb_kwargs = get_default_rb_dict(observation_space.shape, action_space.shape, replay_size)
        self.replay_buffer = ReplayBuffer(**rb_kwargs)

        # Freeze target network
        for p in self.ac.actor_target.parameters():
            p.requires_grad = False

        for p in self.ac.critic_target.parameters():
            p.requires_grad = False

    def load_ac(self, ac):
        self.ac = ac

    def actor_loss(self, obs):
        return - self.ac.critic.q1_forward(obs, self.ac.actor(obs)).mean()

    def critic_loss(self, obs, act, next_obs, rew, done):
        td_error1, td_error2 = self.compute_td_error(obs, act, next_obs, rew, done)

        loss_q1 = huber_loss(td_error1).mean()
        loss_q2 = huber_loss(td_error2).mean()

        loss_q = (loss_q1 + loss_q2).mean()

        return loss_q

    def compute_td_error(self, obs, act, next_obs, rew, done):
        with torch.no_grad():  # target
            pi_target = self.ac.actor_target(next_obs)

            noise = torch.randn_like(pi_target) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_act = torch.clamp(pi_target + noise, -1, 1)

            target_q1, target_q2 = self.ac.critic_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2)
            target = rew + self.gamma * (1 - done) * target_q

        current_q1, current_q2 = self.ac.critic(obs, act)

        td_error1 = current_q1 - target
        td_error2 = current_q2 - target

        return td_error1, td_error2

    def update_targets(self):
        self.ac.actor_target.soft_update(self.ac.actor, self.tau)
        self.ac.critic_target.soft_update(self.ac.critic, self.tau)

    def update_critic(self, obs, act, next_obs, rew, done):
        self.ac.critic_optimizer.zero_grad()
        loss_critic = self.critic_loss(obs, act, next_obs, rew, done)
        loss_critic.backward()
        self.ac.critic_optimizer.step()

        if self.logger is not None:
            self.logger.debug(f'Critic loss: {loss_critic}')

        return loss_critic

    def update_actor(self, obs):
        self.ac.actor_optimizer.zero_grad()
        loss_actor = self.actor_loss(obs)
        loss_actor.backward()
        self.ac.actor_optimizer.step()

        if self.logger is not None:
            self.logger.debug(f'Actor loss: {loss_actor}')

        return loss_actor

    def update_parameters(self, batch_size, timer):

        batch = self.replay_buffer.sample(batch_size)

        # to tensor
        obs = torch.FloatTensor(batch['obs']).to(self.device)
        act = torch.FloatTensor(batch['act']).to(self.device)
        next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
        rew = torch.FloatTensor(batch['rew']).to(self.device)
        done = torch.FloatTensor(batch['done']).to(self.device)

        # update critic
        critic_loss = self.update_critic(obs, act, next_obs, rew, done)
        actor_loss = self.last_actor_loss  # restore value for save

        # update actor & target
        if timer % self.policy_delay == 0:

            actor_loss = self.update_actor(obs)
            self.last_actor_loss = actor_loss

            self.update_targets()

        return actor_loss, critic_loss

    def scale_action(self, act):
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((act - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        low, high = self.action_space.low, self.action_space.high
        if not isinstance(scaled_action, (np.ndarray, np.generic)):
            scaled_action = scaled_action.cpu().detach().numpy()
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def to_tensor(self, obs, dtype=torch.float32):
        obs = torch.as_tensor(obs, dtype=dtype).to(self.device)
        return obs

    def get_random_action_noise(self):
        unscaled_act = np.array([self.action_space.sample()])
        scaled_act = self.scale_action(unscaled_act)

        noise = np.random.normal(0, self.target_noise)
        scaled_act = np.clip(scaled_act + noise, -1, 1)

        return self.unscale_action(scaled_act)

    def get_action_noise(self, obs):
        self.ac.actor.eval()

        act = self.raw_predict(obs)

        # add noise
        noise = np.random.normal(0, self.target_noise)
        scaled_act = np.clip(act + noise, -1, 1)
        # print("Noise:", noise)

        self.ac.actor.train()

        return self.unscale_action(scaled_act)

    def predict(self, obs):
        self.ac.actor.eval()

        act = self.raw_predict(obs)

        self.ac.actor.train()

        return self.unscale_action(act)

    def raw_predict(self, obs):
        self.ac.actor.eval()

        obs = np.expand_dims(obs, axis=0)
        obs = self.to_tensor(obs)
        act = self.ac.act(obs).cpu().data.numpy()

        self.ac.actor.train()

        return act


def eval(env, agent):
    rews, steps = [], []
    for PosIndex in range(1, 1 + 1):
        obs = env.reset(PosIndex=PosIndex, random_position=False)
        done = False
        episode_reward, episode_steps = 0, 0
        while not done:
            act = agent.predict(obs)
            obs, rew, done, _ = env.step(act)
            episode_reward += rew
            episode_steps += 1
        steps.append(episode_steps)
        rews.append(episode_reward)

    return np.mean(rews), np.mean(steps)


def run(steps_per_epoch=4000,
        start_steps=10000,
        update_after=1000,
        n_steps=int(2e5),
        update_every=20,
        policy_delay=2,
        act_noise=0.1,
        target_noise=0.02,
        noise_clip=0.5,
        tau=0.005,
        gamma=0.9,
        actor_lr=1e-3,
        critic_lr=1e-3,
        replay_size=100000,
        batch_size=100,
        seed=100,
        ):

    set_seed_everywhere(seed)
    date = datetime.now().strftime("%b_%d_%Y_%H%M%S")
    logger_main = get_helper_logger('Main', date)
    logger_main.info(f'Process has started')
    logger_main.info(f'step_per_epoch: {steps_per_epoch}')
    logger_main.info(f'start_steps: {start_steps}')
    logger_main.info(f'update_after: {update_after}')
    logger_main.info(f'n_steps: {n_steps}')
    logger_main.info(f'update_every: {update_every}')
    logger_main.info(f'policy_delay: {policy_delay}')
    logger_main.info(f'act_noise: {act_noise}')
    logger_main.info(f'target_noise: {target_noise}')
    logger_main.info(f'noise_clip: {noise_clip}')
    logger_main.info(f'tau: {tau}')
    logger_main.info(f'gamma: {gamma}')
    logger_main.info(f'actor_lr: {actor_lr}')
    logger_main.info(f'critic_lr: {critic_lr}')
    logger_main.info(f'replay_size: {replay_size}')
    logger_main.info(f'batch_size: {batch_size}')
    logger_main.info(f'seed: {seed}')

    env = CrazyCar(renders=True, date=date)
    env = FrameStack(env)
    logger_main.info(f'Environment: {str(env.__class__.__name__)}')
    logger_main.info(f"-----------------")

    agent = TD3(
        observation_space=env.observation_space,
        action_space=env.action_space,
        date=date,
        tau=tau,
        gamma=gamma,
        target_noise=target_noise,
        replay_size=replay_size,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )

    logger = Logger(date)

    logger.save_hyperparameter(
        algorithm='TD3',
        env=env.__class__.__name__,
        observation_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        steps_per_epoch=steps_per_epoch,
        start_steps=start_steps,
        update_after=update_after,
        update_every=update_every,
        act_noise=act_noise,
        policy_delay=policy_delay,
        tau=tau,
        gamma=gamma,
        noise_clip=noise_clip,
        target_noise=target_noise,
        replay_size=replay_size,
        batch_size=batch_size,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        seed=seed
    )

    logger.start()

    best_to_save = float('-inf')
    episode_rew, episode_steps = 0, 0

    obs = env.reset(random_position=False)

    for t in trange(n_steps):

        if t < start_steps:
            unscaled_act = np.array([env.action_space.sample()])
            scaled_act = agent.scale_action(unscaled_act)
        else:
            scaled_act = agent.raw_predict(obs)

        noise = np.random.normal(0, act_noise)
        scaled_act = np.clip(scaled_act + noise, -1, 1)

        next_obs, rew, done, _ = env.step(agent.unscale_action(scaled_act))

        episode_rew += rew
        episode_steps += 1

        agent.replay_buffer.add(obs=obs, next_obs=next_obs, act=scaled_act, rew=rew, done=done)

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
        if t > update_after and t % update_every == 0:
            for j in range(update_every):
                actor_loss, critic_loss = agent.update_parameters(batch_size, j)

                logger.store('Loss/Actor', actor_loss)
                logger.store('Loss/Critic', critic_loss)

        # eval and save
        if (t + 1) % steps_per_epoch == 0:
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
                logger.save_model(agent.ac)

        logger.update_steps()


if __name__ == '__main__':
    run()
