import click
import torch
import math
import numpy as np

from sac_torch.sac import SAC
from pysim.environment import CrazyCar, SingleControl, MultiCar, FrameStack


@click.command()
@click.argument('path')
def main(path):
    env = SingleControl(renders=True, track_id=2)
    env = FrameStack(env)
    model = SAC(env.observation_space, env.action_space)

    actor, critic = torch.load(path)

    model.load_model(actor, critic)

    while True:
        obs = env.reset(random_position=False)
        done = False
        rews = []
        while not done:
            act = model.select_action(obs, evaluate=True)
            # print(act.shape)
            obs, rew, done, _ = env.step(act)

            # print(np.unique(obs))
            rews.append(rew)
            # print(list(obs))
            # print("Reward:", rew)
            # print("Action", act)
        print(np.sum(rews))


if __name__ == '__main__':
    main()
