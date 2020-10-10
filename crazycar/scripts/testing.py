import math
import tensorflow as tf

from absl import app, flags

from crazycar.environments import Environment
from crazycar.agents import SensorAgent
from crazycar.algos_tf import SAC
from crazycar.algos_tf.encoder import Sensor
from crazycar.utils import initial

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_episode", int(1e6), "number of episode for testing")
flags.DEFINE_string("name", None, "experiment name for testing")
flags.mark_flag_as_required("name")


def main(_):

    # initial necessary
    initial()

    # define environment
    env = Environment(map_id=2)
    agents = [SensorAgent]
    positions = [[2.5, 6, math.pi * 2 / 2.0]]
    for agent, pos in zip(agents, positions):
        env.insert_car(agent, pos)

    # define models
    models = SAC(Sensor, 2)
    tmp = tf.saved_model.load(f'./models/{FLAGS.name}/SAC-0')
    models.actor = tmp

    while True:
        done = False
        obs = env.reset()
        while not done:
            # print(obs)
            obs = tf.convert_to_tensor(obs[0]['sensor'])
            act = models.predict({
                "sensor": obs
            })
            obs, rew, done, info = env.step(act)
            done = done[0]

            print(rew)


if __name__ == "__main__":
    app.run(main)

