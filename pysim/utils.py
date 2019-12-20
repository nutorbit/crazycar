from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.ppo2 import PPO2
from stable_baselines.sac import SAC
from stable_baselines.ddpg import DDPG


def get_model(env, name='ppo2', idx_experiment=1):

    name = name.lower()
    
    # Initial Policy Network
    policy = MlpPolicy

    if name == 'ppo2':
        model = PPO2(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )

    if name == 'sac':
        model = SAC(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )
    
    if name == 'ddpg':
        model = DDPG(
            policy=policy,
            env=env,
            verbose=1,
            tensorboard_log=f'./logs/tensorboard/experiment_{idx_experiment}/{name}/'
        )

    return model
