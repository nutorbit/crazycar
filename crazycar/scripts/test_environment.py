import math
import numpy as np

from absl import app

from crazycar.environments import Environment
from crazycar.agents import Racecar, ImageAgent, SensorAgent


def main(args):
    print(args)
    env = Environment()
    env.insert_car(SensorAgent, [2.5, 6, math.pi * 2 / 2.0])
    env.insert_car(ImageAgent, [2.5, 4, math.pi * 2 / 2.0])
    env.insert_car(Racecar, [2.5, 2, math.pi * 2 / 2.0])

    env.reset()

    acts = np.array([[1, 0], [1, 0], [1, 0]]).reshape((3, 2))

    while 1:
        obs, rew, done, info = env.step(acts)


if __name__ == "__main__":
    app.run(main)
