import math
import numpy as np

from absl import app, logging

from crazycar.environments import Environment
from crazycar.agents import Racecar, ImageAgent, SensorAgent


logging.set_verbosity(logging.INFO)
logging.get_absl_handler().setFormatter(None)


def main(_):
    env = Environment(map_id=2)
    # env.insert_car(SensorAgent, [0.2, 0.2, 0])
    env.insert_car(SensorAgent, [2.5, 6, math.pi * 2 / 2.0])
    # env.insert_car(SensorAgent, [2.5, 0.3, math.pi * 2 / 2.0])

    env.reset()
    logging.info("test")

    acts = np.array([[0, 0]]).reshape((1, 2))

    while 1:
        obs, rew, done, info = env.step(acts)
        print(rew)
        # time.sleep(0.5)


if __name__ == "__main__":
    app.run(main)
