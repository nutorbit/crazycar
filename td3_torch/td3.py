import torch
import numpy as np
import notify

from tqdm import trange

from td3_torch.utils import Logger, set_seed, huber_loss, get_default_rb_dict
from td3_torch.policies import ActorCritic, ActorCriticCNN

from pysim.environment import CrazyCar, SingleControl

from cpprb import PrioritizedReplayBuffer
from cpprb import ReplayBuffer


class Agent:
    def __init__(self, observation_space, action_space, logger,
                 polyak=0.995,
                 gamma=0.9,
                 target_noise=0.2,
                 replay_size=100000,
                 noise_clip=0.5,
                 policy_delay=2,
                 name='Anonymous',
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 ):

        self.name = name
        self.observation_space = observation_space
        self.action_space = action_space

        self.polyak = polyak
        self.gamma = gamma
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.logger = logger

        self.ac = ActorCritic(observation_space.shape[0], action_space.shape[0], actor_lr, critic_lr)
        # self.ac = ActorCriticCNN(observation_space.shape[0], action_space.shape[0], actor_lr, critic_lr)

        rb_kwargs = get_default_rb_dict(observation_space.shape, action_space.shape, replay_size)
        # self.replay_buffer = ReplayBuffer(**rb_kwargs)
        self.replay_buffer = PrioritizedReplayBuffer(**rb_kwargs)

        # Freeze target network
        for p in self.ac.actor_target.parameters():
            p.requires_grad = False

        for p in self.ac.critic_target.parameters():
            p.requires_grad = False

    def load_ac(self, ac):
        self.ac = ac

    def actor_loss(self, batch):
        obs = batch['obs']
        return - self.ac.critic.q1_forward(obs, self.ac.actor(obs)).mean()

    def critic_loss(self, batch):
        obs, act, next_obs, rew, done, weights = batch['obs'], batch['act'], batch['next_obs'], batch['rew'], \
                                                 batch['done'], batch['weights']

        # print(obs.shape, act.shape, next_obs.shape, rew.shape, done.shape)

        with torch.no_grad():  # target
            pi_target = self.ac.actor_target(obs)

            noise = torch.randn_like(pi_target) * self.target_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_act = torch.clamp(pi_target + noise, -1, 1)

            target_q1, target_q2 = self.ac.critic_target(obs, next_act)
            target_q = torch.min(target_q1, target_q2)
            target = rew + self.gamma * (1 - done) * target_q

        current_q1, current_q2 = self.ac.critic(obs, act)

        td_error1 = current_q1 - target
        td_error2 = current_q2 - target

        # loss_q1 = (td_error1 ** 2).mean()
        # loss_q2 = (td_error2 ** 2).mean()

        loss_q1 = torch.mean(huber_loss(td_error1) * weights)
        loss_q2 = torch.mean(huber_loss(td_error2) * weights)

        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_td_error(self, obs, act, next_obs, rew, done):
        with torch.no_grad():
            target_q1, target_q2 = self.ac.critic_target(next_obs, self.ac.actor_target(next_obs))
            target_q = torch.min(target_q1, target_q2)
            target_q = rew + ((1 - done) * self.gamma * target_q)

            current_q1, current_q2 = self.ac.critic(obs, act)

            td_error1 = target_q - current_q1
            td_error2 = target_q - current_q2

        return torch.abs(td_error1) + torch.abs(td_error2)

    def update_targets(self):
        self.ac.actor_target.soft_update(self.ac.actor, self.polyak)
        self.ac.critic_target.soft_update(self.ac.critic, self.polyak)

    def update_critic(self, batch):
        self.ac.critic_optimizer.zero_grad()
        loss_critic = self.critic_loss(batch)
        loss_critic.backward()
        self.ac.critic_optimizer.step()

        self.logger.store(f'Loss/loss_critic/{self.name}', loss_critic.mean())

    def update_actor(self, batch):
        self.ac.actor_optimizer.zero_grad()
        loss_actor = self.actor_loss(batch)
        loss_actor.backward()
        self.ac.actor_optimizer.step()

        self.logger.store(f'Loss/loss_actor/{self.name}', loss_actor.mean())

    def update(self, batch, timer):

        self.update_critic(batch)

        if timer % self.policy_delay == 0:

            # Freeze for faster compute
            for p in self.ac.critic.parameters():
                p.requires_grad = False

            self.update_actor(batch)

            # Un-Freeze
            for p in self.ac.critic.parameters():
                p.requires_grad = True

            self.update_targets()

    def scale_action(self, act):
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((act - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action):
        low, high = self.action_space.low, self.action_space.high
        if not isinstance(scaled_action, (np.ndarray, np.generic)):
            scaled_action = scaled_action.cpu().detach().numpy()
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def to_tensor(self, obs, dtype=torch.float32):
        obs = torch.as_tensor(obs, dtype=dtype)
        return obs

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
        obs = self.to_tensor(obs).cuda()
        act = self.ac.act(obs).cpu().data.numpy()

        self.ac.actor.train()

        return act


class TD3:
    def __init__(self, env,
                 steps_per_epoch=4000,
                 start_steps=10000,
                 update_after=1000,
                 update_every=50,
                 policy_delay=2,
                 act_noise=0.3,
                 target_noise=0.3,
                 noise_clip=0.5,
                 polyak=0.995,
                 gamma=0.9,
                 actor_lr=1e-4,
                 critic_lr=1e-4,
                 replay_size=100000,
                 batch_size=200,
                 seed=100):

        # Hyperparameter
        self.env = env
        self.steps_per_epoch = steps_per_epoch
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.act_noise = act_noise
        self.policy_delay = policy_delay
        self.polyak = polyak
        self.gamma = gamma
        self.noise_clip = noise_clip
        self.target_noise = target_noise
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.seed = seed

        self.logger = Logger()

        # Save hyperparameter
        self.logger.save_hyperparameter(
            env=env.__class__.__name__,
            steps_per_epoch=steps_per_epoch,
            start_steps=start_steps,
            update_after=update_after,
            update_every=update_every,
            act_noise=act_noise,
            policy_delay=policy_delay,
            polyak=polyak,
            gamma=gamma,
            noise_clip=noise_clip,
            target_noise=target_noise,
            replay_size=replay_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            seed=seed
        )

        # Set random seed
        # set_seed(seed)

        # Set agent
        self.agent = Agent(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            logger=self.logger,
            polyak=polyak,
            gamma=gamma,
            target_noise=target_noise,
            replay_size=replay_size,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            actor_lr=actor_lr,
            critic_lr=critic_lr
        )

    def eval(self):
        rews, steps = [], []
        for PosIndex in range(1, 11+1):
            obs = self.env.reset(PosIndex=PosIndex, random_position=False)
            done = False
            episode_reward, episode_steps = 0, 0
            while not done:
                act = self.agent.predict(obs)
                # print(act)
                obs, rew, done, _ = self.env.step(act)
                episode_reward += rew
                episode_steps += 1
            steps.append(episode_steps)
            rews.append(episode_reward)

        print(f'[EVALUATION] mean_reward: {np.mean(rews)}, mean_steps: {np.mean(steps)}')

        return np.mean(rews), np.mean(steps)

    def learn(self, epochs):
        total_timesteps = epochs * self.steps_per_epoch

        obs = self.env.reset()
        episode_rew, episode_len, best_rew = 0, 0, float('-inf')
        self.logger.start()

        for t in trange(total_timesteps):
            if t < self.start_steps:
                unscaled_act = np.array([self.env.action_space.sample()])
                scaled_act = self.agent.scale_action(unscaled_act)
            else:
                scaled_act = self.agent.raw_predict(obs)
                # print(scaled_act, scaled_act.shape)

            noise = np.random.normal(0, self.act_noise)
            scaled_act = np.clip(scaled_act + noise, -1, 1)

            # print(scaled_act.shape)
            next_obs, rew, done, _ = self.env.step(self.agent.unscale_action(scaled_act))

            episode_rew += rew
            episode_len += 1

            self.agent.replay_buffer.add(obs=obs, next_obs=next_obs, act=scaled_act, rew=rew, done=done)

            obs = next_obs

            if done:
                obs = self.env.reset()

                self.logger.store('Reward/train', episode_rew)
                self.logger.store('Steps/train', episode_len)

                episode_rew, episode_len = 0, 0

            if t > self.update_after and t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.agent.replay_buffer.sample(self.batch_size)
                    batch = {
                        key: torch.as_tensor(val, dtype=torch.float32).cuda() if key != 'indexes' else val
                        for key, val in batch.items()
                    }
                    # normalize reward
                    batch['rew'] = (batch['rew'] - batch['rew'].mean())/(batch['rew'] + 1e-6)

                    # update
                    self.agent.update(batch, j)

                    # Calculate td error for PER
                    td_error = self.agent.compute_td_error(batch['obs'], batch['act'], batch['next_obs'],
                                                           batch['rew'], batch['done']).cpu().data.numpy()

                    # update priorities
                    self.agent.replay_buffer.update_priorities(batch['indexes'], np.abs(td_error) + 1e-6)

            if (t+1) % self.steps_per_epoch == 0:
                epoch = (t+1)//self.steps_per_epoch

                mean_rew, mean_steps = self.eval()

                self.logger.store('Reward/test', mean_rew)
                self.logger.store('Steps/test', mean_steps)

                # save model here
                if best_rew < mean_steps:
                    best_rew = mean_steps
                    self.logger.save_model(self.agent.ac)

                    # line message
                    notify.alert(f"{self.env.__class__.__name__}\nReward: {mean_rew:.3f}\nSteps:{mean_steps:.3f}\nTimestep: {t+1}/{total_timesteps}")

            self.logger.update_steps()


if __name__ == '__main__':

    print(f"[STATUS] CUDA: {torch.cuda.is_available()}")
    print(f"[STATUS] GPU: {torch.cuda.get_device_name(0)}")

    # torch.cuda.empty_cache()
    # torch.cuda.memory_allocated()
    # torch.set_num_threads(12)

    # env = CrazyCar()
    env = SingleControl()
    # import gym
    # set_seed(100)
    # env = gym.make('MountainCarContinuous-v0')
    model = TD3(env)
    model.learn(1000)
