import math
import numpy as np
import tensorflow as tf

from absl import app, flags

from crazycar.environments import Environment
from crazycar.agents import ImageAgent, SensorAgent
from crazycar.algos import TD3, SAC
from crazycar.encoder import Image, Sensor
from crazycar.utils import evaluation, initial


FLAGS = flags.FLAGS
flags.DEFINE_integer("n_steps", int(1e6), "number of steps for training")
flags.DEFINE_integer("start_steps", 1000, "number of steps for random action")
flags.DEFINE_integer("step_update", 100, "number of steps before update")
flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_integer("eval_steps", int(1e4), "number of steps for evaluation")
flags.DEFINE_string("name", None, "experiment name for logging")
flags.mark_flag_as_required("name")


def main(_):

    # initial necessary
    initial()

    # define environment
    env = Environment(map_id=2)
    agents = [SensorAgent, SensorAgent]
    positions = [[2.4, 1, math.pi / 2], [2.7, 4, math.pi / 2]]
    # positions = [[2.5, 6, math.pi * 2 / 2]]
    for agent, pos in zip(agents, positions):
        env.insert_car(agent, pos)

    # define models
    models = [SAC(Sensor, 1), SAC(Sensor, 1)]
    writers = [
        tf.summary.create_file_writer(f'./logs/{FLAGS.name}/{model.__class__.__name__}-{idx}')
        for idx, model in enumerate(models)
    ]

    step = 0
    max_rew = [float("-inf") for _ in range(len(models))]

    while step < FLAGS.n_steps:
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
            if step > FLAGS.batch_size and step % FLAGS.step_update == 0:
                for _ in range(FLAGS.step_update):
                    for idx, model in enumerate(models):
                        metric = model.update_params(step, FLAGS.batch_size)

                        # write tensorboard
                        with writers[idx].as_default():
                            model.write_metric(metric, step)

            # evaluation
            if step % FLAGS.eval_steps == 0:
                mean_rew, mean_step = evaluation(env, models)
                print(f"|Evaluation at {step:08d}| Mean reward: {str(mean_rew)}, Mean step: {str(mean_step)}")

                for idx, model in enumerate(models):

                    # save model
                    if mean_rew[idx] > max_rew[idx]:
                        # tf.saved_model.save(model.actor, f"./models/{FLAGS.name}/{model.__class__.__name__}-{idx}")
                        max_rew[idx] = mean_rew[idx]

                    # addition metric
                    with writers[idx].as_default():
                        tf.summary.scalar("track/mean_reward", mean_rew[idx], step)
                        tf.summary.scalar("track/mean_step", mean_step, step)

            # to next state
            step += 1
            done = done[0]
            obs = next_obs


if __name__ == "__main__":
    app.run(main)
