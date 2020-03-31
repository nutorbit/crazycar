import threading
import torch
import cloudpickle
import notify

import numpy as np

from pysim.environment import SingleControl

from td3_torch.utils import get_default_rb_dict
from td3_torch.td3 import TD3

from multiprocessing import Process, Queue, Value, Event, Lock, cpu_count
from multiprocessing.managers import SyncManager

from cpprb import ReplayBuffer, PrioritizedReplayBuffer


class Apex:
    """
    Ref: https://arxiv.org/pdf/1803.00933.pdf
    """

    def __init__(self, env_fn, total_steps):
        self.env_fn = env_fn
        self.sample_env = env_fn()
        self.total_steps = total_steps
        # self.n_explorer = cpu_count() - 2
        self.n_explorer = 10

        # Create manager to share PER between a learner and explorers.
        SyncManager.register('PrioritizedReplayBuffer', PrioritizedReplayBuffer)

        manager = SyncManager()
        manager.start()

        # get observation & action space
        self.observation_space = self.sample_env.observation_space
        self.action_space = self.sample_env.action_space

        rb_kwargs = get_default_rb_dict(self.observation_space.shape, self.action_space.shape, int(1e6))
        rb_kwargs["check_for_update"] = True
        self.global_rb = manager.PrioritizedReplayBuffer(**rb_kwargs)

        self.queues = [manager.Queue() for _ in range(self.n_explorer + 1)]

        self.is_training_done = Event()

        self.lock = manager.Lock()

        self.trained_steps = Value('i', 0)

    @staticmethod
    def policy_fn(env, device='cpu'):
        return TD3(
            env=env,
            device=device
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

    @staticmethod
    def explorer(name, global_rb, queue, is_training_done,
                 lock, env_fn, policy_fn, set_weights_fn):

        print(f"Explore {name}: Starting...")

        env_fn = cloudpickle.loads(env_fn)
        policy_fn = cloudpickle.loads(policy_fn)
        set_weights_fn = cloudpickle.loads(set_weights_fn)

        env = env_fn()

        policy = policy_fn(
            env=env,
            device='cpu'
        )

        policy.agent.name = name

        rb_kwargs = get_default_rb_dict(env.observation_space.shape, env.action_space.shape, 2000)
        local_rb = ReplayBuffer(**rb_kwargs)

        n_sample, n_sample_old = 0, 0

        obs = env.reset()
        total_reward = 0.
        total_rewards = []

        while not is_training_done.is_set():
            n_sample += 1

            if n_sample < policy.start_steps:
                act = policy.agent.get_random_action_noise()
            else:
                act = policy.agent.get_action_noise(obs)

            next_obs, rew, done, _ = env.step(act)

            local_rb.add(obs=obs, act=act, next_obs=next_obs, rew=rew, done=done)

            total_reward += rew
            obs = next_obs

            if done:
                obs = env.reset()
                total_rewards.append(total_reward)
                total_reward = 0

            if not queue.empty():
                set_weights_fn(policy, queue.get())

            # Add collected experiences to global
            if local_rb.get_stored_size() == 2000:
                samples = local_rb.sample(2000)

                # transform to tensor
                obs_ts = torch.as_tensor(samples['obs'], dtype=torch.float32)
                act_ts = torch.as_tensor(samples['act'], dtype=torch.float32)
                next_obs_ts = torch.as_tensor(samples['next_obs'], dtype=torch.float32)
                rew_ts = torch.as_tensor(samples['rew'], dtype=torch.float32)
                done_ts = torch.as_tensor(samples['done'], dtype=torch.float32)

                td_errors = policy.agent.compute_td_error(obs_ts, act_ts, next_obs_ts, rew_ts, done_ts).cpu().numpy()

                lock.acquire()

                global_rb.add(
                    obs=samples['obs'],
                    act=samples['act'],
                    next_obs=samples['next_obs'],
                    rew=samples['rew'],
                    done=samples['done'],
                    priorities=td_errors + 1e-6
                )

                lock.release()

                local_rb.clear()

    @staticmethod
    def learner(name, global_rb, trained_steps, is_training_done,
                lock, env_fn, policy_fn, get_ac_fn, n_training, queues, device):
        print(f"Learner {name}: Starting...")

        env_fn = cloudpickle.loads(env_fn)
        policy_fn = cloudpickle.loads(policy_fn)
        get_ac_fn = cloudpickle.loads(get_ac_fn)

        env = env_fn()

        policy = policy_fn(
            env=env,
            device=device
        )

        policy.agent.name = name
        policy.agent.logger = None

        while not is_training_done.is_set() and global_rb.get_stored_size() < 10000:
            # print(global_rb.get_stored_size())
            continue

        while not is_training_done.is_set():
            trained_steps.value += 1

            lock.acquire()

            batch = global_rb.sample(policy.batch_size)

            lock.release()

            # transform
            batch = {
                key: torch.as_tensor(val, dtype=torch.float32).to(device) if key != 'indexes' else val
                for key, val in batch.items()
            }

            # normalize reward
            batch['rew'] = (batch['rew'] - batch['rew'].mean()) / (batch['rew'] + 1e-6)

            # update
            policy.agent.update(batch, trained_steps.value)

            # td-error
            td_error = policy.agent.compute_td_error(batch['obs'], batch['act'], batch['next_obs'],
                                                     batch['rew'], batch['done']).cpu().numpy()

            global_rb.update_priorities(batch['indexes'], np.abs(td_error) + 1e-6)

            ac = get_ac_fn(policy)

            # broadcast to update each explorer
            if trained_steps.value % policy.update_every == 0:
                for i in range(len(queues) - 1):
                    queues[i].put(ac)

            # evaluation
            if trained_steps.value % 100 == 0:
                # print(trained_steps.value)
                queues[-1].put(ac)
                queues[-1].put(trained_steps.value)

            if trained_steps.value >= n_training:
                is_training_done.set()

    @staticmethod
    def evaluator(name, is_training_done, env_fn, policy_fn, set_weights_fn, n_training, queue):
        print(f"Evaluator {name}: Starting...")

        env_fn = cloudpickle.loads(env_fn)
        policy_fn = cloudpickle.loads(policy_fn)
        set_weights_fn = cloudpickle.loads(set_weights_fn)

        env = env_fn()

        policy = policy_fn(
            env=env,
            device='cpu'
        )

        policy.agent.name = name
        policy.logger.start()

        best_mean_steps, best_mean_rews = float('-inf'), float('-inf')

        while not is_training_done.is_set():
            if queue.empty():
                continue
            else:
                ac = queue.get()
                trained_steps = queue.get()

                set_weights_fn(policy, ac)

                rews, steps = [], []

                for PosIndex in range(1, 11 + 1):
                    obs = env.reset(PosIndex=PosIndex, random_position=False)
                    done = False
                    episode_reward, episode_steps = 0, 0
                    while not done:
                        act = policy.agent.predict(obs)
                        # print(act)
                        obs, rew, done, _ = env.step(act)
                        episode_reward += rew
                        episode_steps += 1
                    steps.append(episode_steps)
                    rews.append(episode_reward)

                mean_rews = np.mean(rews)
                mean_steps = np.mean(steps)

                policy.logger.store("Reward/Evaluator", mean_rews)
                policy.logger.store("Steps/Evaluator", mean_steps)

                print(f'({trained_steps:07d})[EVALUATION] mean_reward: {mean_rews}, mean_steps: {mean_steps}')

                if best_mean_steps < mean_steps:
                    best_mean_steps = mean_steps

                    # save model
                    policy.logger.save_model(policy.agent.ac)

                    # line message
                    notify.alert(f"{env.__class__.__name__} (Ape-X)\nReward: {mean_rews:.3f}\nSteps: {mean_steps:.3f}\nTimestep: {trained_steps}/{n_training}")

                policy.logger.update_steps()

    def run(self):

        tasks = []

        # add explorer
        for i in range(self.n_explorer):
            tasks.append(
                Process(
                    target=self.explorer,
                    args=(f'Explorer_{i}', self.global_rb, self.queues[i], self.is_training_done, self.lock,
                          cloudpickle.dumps(self.env_fn), cloudpickle.dumps(self.policy_fn),
                          cloudpickle.dumps(self.set_weights_fn))
                )
            )

        # add learner
        tasks.append(
            Process(
                target=self.learner,
                args=('Learner', self.global_rb, self.trained_steps, self.is_training_done, self.lock,
                      cloudpickle.dumps(self.env_fn), cloudpickle.dumps(self.policy_fn),
                      cloudpickle.dumps(self.get_ac_fn), self.total_steps, self.queues, 'cpu')
            )
        )

        # add evaluator
        tasks.append(
            Process(
                target=self.evaluator,
                args=('Evaluator', self.is_training_done, cloudpickle.dumps(self.env_fn),
                      cloudpickle.dumps(self.policy_fn), cloudpickle.dumps(self.set_weights_fn), self.total_steps, self.queues[-1])
            )
        )

        for task in tasks:
            task.start()
        for task in tasks:
            task.join()


def main():
    runner = Apex(
        env_fn=lambda: SingleControl(),
        total_steps=int(1e6),
    )
    runner.run()


if __name__ == "__main__":
    # TODO: support CUDA for training
    # envs = MultiThreadEnv(lambda: SingleControl(), 512, 4)
    # obs = envs.py_reset()
    # print(obs.shape)
    torch.multiprocessing.set_start_method('spawn')
    main()
