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


def get_model(name='ppo2', idx_experiment=1):

    name = name.lower()
    
    env = CrazyCar(renders=False)

    if name == 'ppo1':
        from stable_baselines.common.policies import MlpPolicy

        policy = MlpPolicy

        # env = SubprocVecEnv([lambda: CrazycarGymEnv4(renders=False, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP) for _ in range(N_PARALLEL)])
        env = Monitor(env, LOG_DIR, allow_early_resets=True)


        model = PPO1(
            policy=policy,
            env=env,
            # n_steps=1024,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )

    if name == 'ppo2':
        from stable_baselines.common.policies import MlpPolicy

        policy = MlpPolicy

        env = SubprocVecEnv([lambda: CrazyCar(renders=False) for _ in range(N_PARALLEL)])
        # env = Monitor(env, LOG_DIR, allow_early_resets=True)


        model = PPO2(
            policy=policy,
            env=env,
            # n_steps=1<<10,
            # nminibatches=1<<7,
            # cliprange=0.1,
            # learning_rate=0.0001,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )

    if name == 'sac':

        from stable_baselines.sac import MlpPolicy

        policy = MlpPolicy

        env = Monitor(env, LOG_DIR, allow_early_resets=True)

        model = SAC(
            policy=policy,
            env=env,
            # buffer_size=1e6,
            # gamma=0.9975,
            # learning_starts=1<9,
            # batch_size=1<<9,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/',
        )
    
    if name == 'ddpg':

        from stable_baselines.ddpg import MlpPolicy

        policy = MlpPolicy

        env = Monitor(env, LOG_DIR, allow_early_resets=True)

        model = DDPG(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )

    if name == 'td3':

        from stable_baselines.td3 import MlpPolicy

        policy = MlpPolicy

        env = Monitor(env, LOG_DIR, allow_early_resets=True)

        model = TD3(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )

    return model
