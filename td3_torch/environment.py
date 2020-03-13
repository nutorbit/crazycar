import threading
import torch

import numpy as np


class MultiThreadEnv(object):

    def __init__(self, env_fn, batch_size, thread_pool=4, max_episode_steps=1000):

        self.batch_size = batch_size
        self.thread_pool = thread_pool
        self.batch_thread = batch_size // thread_pool
        self.envs = [env_fn() for _ in range(batch_size)]

        # collects environment information
        sample_env = env_fn()
        sample_obs = sample_env.reset()
        self._sample_env = sample_env
        self.observation_shape = sample_obs.shape[0]

        # episode time limit
        self.max_episode_steps = max_episode_steps

        self.list_obs = [None] * self.batch_size
        self.list_rewards = [None] * self.batch_size
        self.list_done = [None] * self.batch_size
        self.list_steps = [0] * self.batch_size

        self.py_reset()

    @property
    def original_env(self):
        return self._sample_env

    def step(self, actions):

        obs, rew, done = self.py_step(actions)

        obs = obs.reshape((self.batch_size, self.observation_shape))
        rew = rew.reshape((self.batch_size, 1))
        done = done.reshape((self.batch_size, 1))

        return obs, rew, done, None

    def py_step(self, actions):

        def _process(offset):
            for idx_env in range(offset, offset+self.batch_thread):
                next_obs, rew, done, _ = self.envs[idx_env].step(actions[idx_env])
                self.list_obs[idx_env] = next_obs
                self.list_rewards[idx_env] = rew
                self.list_done[idx_env] = done
                self.list_steps[idx_env] += 1

        threads = []
        for i in range(self.thread_pool):
            thread = threading.Thread(
                target=_process, args=[i*self.batch_thread])
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        for i in range(self.batch_size):
            if self.list_steps[i] == self.max_episode_steps:
                self.list_done[i] = False

        obs = np.stack(self.list_obs, axis=0)
        rew = np.stack(self.list_rewards, axis=0).astype(np.float32)
        done = np.stack(self.list_done, axis=0).astype(np.float32)

        for i in range(self.batch_size):
            if self.list_done[i] or self.list_steps[i] == self.max_episode_steps:
                self.list_obs[i] = self.envs[i].reset()
                self.list_steps[i] = 0

        return obs, rew, done

    def py_observation(self):
        obs = np.stack(self.list_obs, axis=0).astype(np.float32)
        return obs

    def py_reset(self):
        for idx_env, env in enumerate(self.envs):
            obs = env.reset()
            self.list_obs[idx_env] = obs.astype(np.float32)

        return np.stack(self.list_obs, axis=0)

    @property
    def max_action(self):
        return float(self._sample_env.action_space.high[0])

    @property
    def min_action(self):
        return float(self._sample_env.action_space.low[0])

    @property
    def state_dim(self):
        return self._sample_env.observation_space.shape[0]