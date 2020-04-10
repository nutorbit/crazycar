import click
import torch
import time
import numpy as np

from sac_torch.sac import SAC
from pysim.environment import CrazyCar, SingleControl, MultiCar


@click.command()
@click.option('--path1', default='./models/Apr_07_2020_222742/td3_32000.pth')
@click.option('--path2', default='./models/Apr_07_2020_224827/td3_192000.pth')
def main(path1, path2):
    env = MultiCar(renders=True)
    model1 = SAC(env.observation_space.shape[0], env.action_space)
    actor1, critic1 = torch.load(path1)
    model1.load_model(actor1, critic1)

    model2 = SAC(env.observation_space.shape[0], env.action_space)
    actor2, critic2 = torch.load(path2)
    model2.load_model(actor2, critic2)

    while True:
        obs = env.reset()
        done = False
        rews = []
        while not done:
            act1 = model1.select_action(obs[0], evaluate=True)
            act2 = model2.select_action(obs[1], evaluate=True)
            # print(act1.shape, act2.shape)
            if act1.shape == (1, ):
                act1 = np.array([1, act1[0]])
            if act2.shape == (1, ):
                act2 = np.array([1, act2[0]])
            act = [act1, act2]

            obs, rew, done, _ = env.step(act)
            # print(done)
            # print('car1', obs[0][3])
            # print('car2', obs[1][3])
            print('car front', np.unique(obs[0]))
            print('car tail', np.unique(obs[1]))
            print(rew)
            rews.append(rew)
            # time.sleep(0.1)
            # print("Reward:", rew)
            # print("Action", act)
        print(np.sum(rews))


if __name__ == '__main__':
    main()
