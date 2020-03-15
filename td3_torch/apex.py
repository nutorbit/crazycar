import multiprocessing
import threading
import torch

import numpy as np

from pysim.environment import SingleControl

from td3_torch.environment import MultiThreadEnv
from td3_torch.utils import get_default_rb_dict
from td3_torch.td3 import TD3

from multiprocessing import Process, Queue, Value, Event, Lock
from multiprocessing.managers import SyncManager

from cpprb import ReplayBuffer, PrioritizedReplayBuffer


class Apex:
    """
    Ref: https://arxiv.org/pdf/1803.00933.pdf
    """

    def __init__(self, env_fn, batch_size, thread_pool):
        self.env_fn = env_fn
        self.sample_env = env_fn()
        self.env = env_fn()
        self.batch_size = batch_size
        self.thread_pool = thread_pool

        # Create manager to share PER between a learner and explorers.
        SyncManager.register('PrioritizedReplayBuffer', PrioritizedReplayBuffer)

        manager = SyncManager()
        manager.start()

        # get observation & action space
        self.observation_space = self.sample_env.observation_space
        self.action_space = self.sample_env.action_space

        rb_kwargs = get_default_rb_dict(self.observation_space.shape[0], self.action_space.shape[0], int(1e6))
        rb_kwargs["check_for_update"] = True
        self.global_rb = manager.PrioritizedReplayBuffer(**rb_kwargs)

        self.queues = [manager.Queue() for _ in range(2)]

        self.is_training_done = Event()

        self.lock = manager.Lock()

        self.trained_steps = Value('i', 0)

    @staticmethod
    def policy_fn(env, replay_size=int(1e6)):
        return TD3(
            env=env,
            replay_size=replay_size
        )

    @staticmethod
    def get_ac_fn(policy):
        return [
            policy.agent.ac.actor,
            policy.agent.ac.actor_target,
            policy.agent.ac.critic,
            policy.agent.ac.critic_target
        ]

    @staticmethod
    def set_weights_fn(policy, ac):
        actor, actor_target, critic, critic_target = ac

        policy.agent.ac.actor.hard_update(actor)
        policy.agent.ac.actor_target.hard_update(actor_target)
        policy.agent.ac.critic.hard_update(critic)
        policy.agent.ac.critic_target.hard_update(critic_target)

    def explorer(self):
        envs = MultiThreadEnv(
            env_fn=self.env_fn,
            batch_size=self.batch_size,
            thread_pool=self.thread_pool
        )
        env = envs._sample_env

        policy = self.policy_fn(
            env=env
        )

        rb_kwargs = get_default_rb_dict(self.observation_space.shape[0], self.action_space.shape[0], 1024)
        rb_kwargs["env_dict"]["priorities"] = {}
        local_rb = ReplayBuffer(**rb_kwargs)
        local_idx = np.arange(1024)

        n_sample, n_sample_old = 0, 0

        while not self.is_training_done.is_set():
            n_sample += self.batch_size
            obses = envs.py_observation()
            actions = policy.agent.get_action_noise(obses)[0]
            # print(actions, actions.shape)
            next_obses, rews, dones, _ = envs.step(actions)

            # transform to tensor
            obses = torch.as_tensor(obses, dtype=torch.float32).cuda()
            actions = torch.as_tensor(actions, dtype=torch.float32).cuda()
            next_obses = torch.as_tensor(next_obses, dtype=torch.float32).cuda()
            rews = torch.as_tensor(rews, dtype=torch.float32).cuda()
            dones = torch.as_tensor(dones, dtype=torch.float32).cuda()

            td_errors = policy.agent.compute_td_error(obses, actions, next_obses, rews, dones).cpu().numpy()

            local_rb.add(obs=obses, act=actions, next_obs=next_obses, rew=rews, done=dones, priorities=td_errors + 1e-6)

            if not self.queues[0].empty():
                self.set_weights_fn(policy, self.queues[0].get())

            # Add collected experiences to global
            if local_rb.get_stored_size() == 1024:
                samples = local_rb._encode_sample(local_idx)
                priorities = np.squeeze(samples["priorities"])
                self.lock.acquire()
                self.global_rb.add(
                    obs=samples["obs"], act=samples["act"], rew=samples["rew"],
                    next_obs=samples["next_obs"], done=samples["done"],
                    priorities=priorities
                )
                self.lock.release()
                local_rb.clear()
                n_sample_old = n_sample

    def learner(self):

        policy = self.policy_fn(
            env=self.env
        )

        while not self.is_training_done.is_set() and self.global_rb.get_stored_size() < 1000:
            continue

        while not self.is_training_done.is_set():
            self.trained_steps.value += 1
            self.lock.acquire()

            batch = self.global_rb.sample(self.batch_size)
            batch = {key: torch.as_tensor(val, dtype=torch.float32).cuda() for key, val in batch.items()}

            self.lock.release()

            # td_error


if __name__ == "__main__":

    # envs = MultiThreadEnv(lambda: SingleControl(), 512, 4)
    # obs = envs.py_reset()
    # print(obs.shape)

    runner = Apex(
        env_fn=lambda: SingleControl(),
        batch_size=500,
        thread_pool=4
    )
    runner.explorer()
