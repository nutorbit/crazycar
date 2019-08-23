import click

from pysim import CrazycarGymEnv4
from pysim.constants import *

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines.sac import SAC
from stable_baselines.ddpg import DDPG


@click.command()
@click.option('--load', help='load model for continue train')
@click.option('--nupdate', default=100, help='number of update')
@click.argument('output')
def main(load, nupdate, output):

    # init environment
    env = CrazycarGymEnv4(renders=False, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP)
    env = SubprocVecEnv([lambda: env for _ in range(N_PARALLEL)])
    # env = DummyVecEnv([lambda: env])

    # print(env.action_space.low, env.action_space.high)

    # init policy
    policy = MlpPolicy

    if load:
        model = PPO2.load(load, env, 
                        n_steps=N_STEPS,
                        nminibatches=N_MINIBATCHES,
                        lam=LAMBDA,
                        gamma=GAMMA,
                        verbose=1,
                        tensorboard_log='./logs/tensorboard/ppo2/'
        )

    else:
        model = PPO2(policy, env,
                    n_steps=N_STEPS,
                    nminibatches=N_MINIBATCHES,
                    lam=LAMBDA,
                    gamma=GAMMA,
                    verbose=1,
                    tensorboard_log='./logs/tensorboard/ppo2/'
        )
        # model = DDPG("MlpPolicy", env, verbose=1)

    iters = (N_STEPS * N_PARALLEL) * nupdate

    # learn
    model.learn(total_timesteps=iters)

    # save
    model.save(f'./pysim/models/{output}.pkl')


if __name__ == '__main__':
	main()