import math
import numpy as np
import tensorflow as tf

from absl import app, flags

from crazycar.environments import Environment
from crazycar.agents import ImageAgent, SensorAgent
from crazycar.algos import TD3, SAC
from crazycar.encoder import Image, Sensor
from crazycar.utils import set_seed


FLAGS = flags.FLAGS
flags.DEFINE_integer("n_episode", 100, "number of episode")
flags.DEFINE_integer("start_steps", 1000, "number of step for random action")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_string("name", None, "experiment name for logging")

flags.mark_flag_as_required("name")


def main(_):
    set_seed()
    env = Environment(map_id=2)
    agents = [SensorAgent]
    positions = [[2.9 - 0.7 / 2, 1.1, math.pi / 2]]
    for agent, pos in zip(agents, positions):
        env.insert_car(agent, pos)

    models = [SAC(Sensor, 2)]
    writer = tf.summary.create_file_writer(f'./logs/{FLAGS.name}/')
    step = 0

    for _ in range(FLAGS.n_episode):
        obs = env.reset()
        done = False

        while not done:

            # get action
            acts = []
            for idx in range(len(models)):
                if step > FLAGS.start_steps:
                    acts.append(models[idx].predict(obs[idx]))
                else:
                    acts.append(models[idx].random_action())
            acts = np.squeeze(np.array(acts), axis=1)

            # apply action
            next_obs, rew, done, info = env.step(acts)
            print(acts, rew)

            # save replay
            for idx in range(len(models)):
                models[idx].rb.store({
                    "obs": obs[idx],
                    "act": acts[idx],
                    "next_obs": next_obs[idx],
                    "rew": rew[idx],
                    "done": done
                })

            # update params
            if step > FLAGS.batch_size:
                for idx, model in enumerate(models, start=1):
                    metric = model.update_params(step, FLAGS.batch_size)

                    # write tensorboard
                    with writer.as_default():
                        model.write_metric(metric, step, idx)

            # to next state
            step += 1
            done = done[0]
            obs = next_obs


if __name__ == "__main__":
    app.run(main)

