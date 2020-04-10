import click
import torch
import math
import numpy as np

from sac_torch.sac import SAC
from pysim.environment import CrazyCar, SingleControl, MultiCar


@click.command()
@click.option('--path', default='./models/Mar_06_2020_122846/td3_284000.pth')
def main(path):
    env = CrazyCar(renders=True)
    model = SAC(env.observation_space.shape[0], env.action_space)

    actor, critic = torch.load(path)

    model.load_model(actor, critic)

    while True:
        obs = env.reset(random_position=False, newCarPos=[2.9 - 0.7/2, 1.1, math.pi/2])
        done = False
        rews = []
        while not done:
            act = model.select_action(obs, evaluate=True)
            # print(act.shape)
            obs, rew, done, _ = env.step(act)

            # print(np.unique(obs))
            rews.append(rew)
            # print(list(obs))
            print("Reward:", rew)
            print("Action", act)
        print(np.sum(rews))


if __name__ == '__main__':
    main()
