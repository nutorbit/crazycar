# add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym

# from pysim.crazycarGymEnv import CrazycarGymEnv
from . import CrazycarGymEnv
from . import CrazycarGymEnv3

from baselines import deepq


def main(modelfile="racecar_model_test.pkl"):
    env = CrazycarGymEnv3(renders=True, isDiscrete=True)
    print(modelfile)
    act = deepq.load(modelfile)
    while True:
        obs, done = env.reset(), False
        print("===================================")
        print("obs")
        print(obs)
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    if (len(sys.argv) != 2):
        main()
    else:
        main(sys.argv[1])
