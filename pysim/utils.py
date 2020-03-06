import tensorflow as tf
import numpy as np

from stable_baselines.ppo2 import PPO2
from stable_baselines.sac import SAC
from stable_baselines.ddpg import DDPG
from stable_baselines.td3 import TD3
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.bench import Monitor

from pysim.constants import *
from pysim.environment import CrazyCar


def get_model(t, name='ppo2', idx_experiment=1):

    name = name.lower()
    
    env = CrazyCar(renders=False)

    log_path = f'./logs/tensorboard/experiment_{idx_experiment}/{name}/{t}/'

    if name == 'ppo1':
        from stable_baselines.common.policies import MlpPolicy

        policy = MlpPolicy

        # env = SubprocVecEnv([lambda: CrazycarGymEnv4(renders=False, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP) for _ in range(N_PARALLEL)])
        env = Monitor(env, LOG_DIR, allow_early_resets=True)


        model = PPO1(
            policy=policy,
            env=env,
            # n_steps=1024,
            gamma=0.9,
            # verbose=1,
            tensorboard_log=log_path
        )

    if name == 'ppo2':
        from stable_baselines.common.policies import MlpPolicy

        policy = MlpPolicy

        policy_kwargs = dict(act_fun=tf.nn.relu, net_arch=[256, 128, 64])

        env = SubprocVecEnv([lambda: CrazyCar(renders=False) for _ in range(N_PARALLEL)])
        # env = Monitor(env, LOG_DIR, allow_early_resets=True)


        model = PPO2(
            policy=policy,
            env=env,
            # n_steps=1<<10,
            # nminibatches=1<<10,
            # gamma=0.9,
            # ent_coef=0,
            # learning_rate=0.00003,
            verbose=1,
            tensorboard_log=log_path,
            policy_kwargs=policy_kwargs
        )

    if name == 'sac':

        from stable_baselines.sac import MlpPolicy

        policy = MlpPolicy

        # env = VecNormalize(DummyVecEnv([lambda: Monitor(env, LOG_DIR, allow_early_resets=True)]))
        env = Monitor(env, LOG_DIR, allow_early_resets=True)

        model = SAC(
            policy=policy,
            env=env,
            # ent_coef='auto_0.1',
            buffer_size=1000000,
            # gamma=0.9,
            # learning_starts=1<9,
            learning_rate=10000,
            train_freq=1,
            # batch_size=1<<9,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=log_path,
        )
    
    if name == 'ddpg':

        from stable_baselines.ddpg import MlpPolicy

        policy = MlpPolicy

        # env = DummyVecEnv([lambda: env, lambda: env])

        env = Monitor(env, LOG_DIR, allow_early_resets=True)

        model = DDPG(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=log_path
        )

    if name == 'td3_tf':

        from stable_baselines.td3 import MlpPolicy
        from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

        policy = MlpPolicy

        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))

        # env = DummyVecEnv([lambda: env, lambda: env])

        env = Monitor(env, LOG_DIR, allow_early_resets=True)

        model = TD3(
            policy=policy,
            env=env,
            action_noise=action_noise,
            buffer_size=1000000,
            gradient_steps=1000,
            learning_starts=10000,
            train_freq=1000,
            policy_kwargs=dict(layers=[400, 300]),
            tensorboard_log=log_path,
            verbose=1,
        )

    return model
