import click
import torch
import numpy as np

from td3_torch.td3 import TD3
from pysim.environment import CrazyCar, SingleControl


@click.command()
@click.option('--path', default='./models/Mar_06_2020_122846/td3_284000.pth')
def main(path):
    env = SingleControl(renders=True)
    model = TD3(env)

    # ac = torch.load('./models/Feb_14_2020_201020/td3_380000.pth')
    # ac = torch.load('./models/Feb_15_2020_190815/td3_1456000.pth')
    # ac = torch.load('./models/Feb_12_2020_213156/td3_420000.pth')
    # ac = torch.load('./models/Mar_06_2020_122846/td3_284000.pth')
    ac = torch.load(path)

    model.agent.load_ac(ac)

    while True:
        obs = env.reset(random_position=False)
        done = False
        rews = []
        while not done:
            act = model.agent.predict(obs)
            # print(act.shape)
            obs, rew, done, _ = env.step(act)
            # print(obs)
            rews.append(rew)
            print("Reward:", rew)
            print("Action", act)
        print(np.sum(rews))


if __name__ == '__main__':
    main()
