import math
import numpy as np
import tensorflow as tf

from absl import app

from crazycar.environments import Environment
from crazycar.agents import ImageAgent, SensorAgent
from crazycar.algos import DDPG
from crazycar.encoder import Image, Sensor


def rollout(env):
    """
    Rollout environment

    Args:
        env: environment object
    """

    models = [DDPG(Sensor, 2), DDPG(Sensor, 2)]
    writer = tf.summary.create_file_writer('./logs/test')

    obs = env.reset()
    done = False
    step = 0

    while not done:

        # get action
        acts = []
        for idx in range(len(models)):
            acts.append(models[idx].predict(obs[idx]))
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
        if step > 128:
            # print("Yes")
            for idx, model in enumerate(models, start=1):
                loss = model.update_params(step)

                with writer.as_default():
                    tf.summary.scalar(f"loss/actor_{idx}", loss['actor_loss'], step)
                    tf.summary.scalar(f"loss/critic_{idx}", loss['critic_loss'], step)

        step += 1
        done = done[0]
        obs = next_obs


def main(_):
    env = Environment()
    agents = [SensorAgent, SensorAgent]
    positions = [[2.5, 6, math.pi * 2 / 2.0], [2.5, 4, math.pi * 2 / 2.0]]
    for agent, pos in zip(agents, positions):
        env.insert_car(agent, pos)
    rollout(env)


if __name__ == "__main__":
    app.run(main)

