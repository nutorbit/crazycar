import numpy as np
import tensorflow as tf
import os
import json

from datetime import datetime


class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size=100000):
        self.obs = np.zeros((size, *obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act = np.zeros((size, *act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.cap = size

    def add(self, obs, next_obs, act, rew, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size=32):
        idx = np.random.randint(low=0, high=self.size, size=batch_size)
        batch = dict(
            obs=self.obs[idx],
            next_obs=self.next_obs[idx],
            act=self.act[idx],
            rew=self.rew[idx],
            done=self.done[idx]
        )
        return {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in batch.items()}


class Logger:

    def __init__(self):
        self.time = datetime.now()
        self.start_date = self.time.strftime("%b_%d_%Y_%H%M%S")
        self.steps = 0
        self.save_hy = False
        self.writer = None
        self.hyperparameter = None

    def start(self):

        # Create lops directory
        if not os.path.exists(f'./logs/{self.start_date}'):
            os.makedirs(f'./logs/{self.start_date}')

        self.writer = tf.summary.create_file_writer(f'./logs/{self.start_date}/')

        # Create model directory
        if not os.path.exists(f'./models/{self.start_date}'):
            os.makedirs(f'./models/{self.start_date}')

        with open(f'./logs/{self.start_date}/params.json', 'w') as f:
            json.dump(self.hyperparameter, f)

    def save_hyperparameter(self, **kwargs):

        # Save hyperparameter
        self.hyperparameter = kwargs

    def update_steps(self):
        self.steps += 1

    def save_model(self, model):
        actor = model.actor
        actor_target = model.actor_target

        critic = model.critic
        critic_target = model.critic_target

        # tf.saved_model.save(model, f'./models/{self.start_date}/td3_{self.steps + 1}.h5')

        actor.save_weights(f'./models/{self.start_date}/actor_{self.steps + 1}')
        actor_target.save_weights(f'./models/{self.start_date}/actor_target_{self.steps + 1}')

        critic.save_weights(f'./models/{self.start_date}/critic_{self.steps + 1}')
        critic_target.save_weights(f'./models/{self.start_date}/critic_target_{self.steps + 1}')

    def store(self, name, val):
        with self.writer.as_default():
            tf.summary.scalar(name, np.mean(val), self.steps)
