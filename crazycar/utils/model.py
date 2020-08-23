import random
import tensorflow as tf
import numpy as np

from collections import deque
from tensorflow.keras import layers


def set_seed(seed=100):
    """
    Set global seed

    Args:
        seed: seed number
    """

    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_mlp(sizes, activation, output_activation=None):
    """
    Create MLP

    Args:
        sizes: unit size for each layer
        activation: activation for apply each layer except last layer
        output_activation: activation for last layer

    Returns:
        layer block
    """

    l = [layers.Input(sizes[0])]
    for i in range(1, len(sizes)):
        if i != len(sizes) - 1:
            l.append(layers.Dense(sizes[i], activation=activation))
        else:
            l.append(layers.Dense(sizes[i], activation=output_activation))
    return tf.keras.Sequential(l)


class Replay:
    """
    Experience replay for RL
    """

    def __init__(self, maxlen=int(1e5)):
        self.data = deque(maxlen=maxlen)

    def store(self, dict_data):
        """
        Store data

        Args:
            dict_data:
            {
                "obs": xxx
                "act": xxx
                "next_obs": xxx
                "rew": xxx
                "done": xxx
            }
        """

        self.data.append(dict_data)

    def process(self, data):
        """
        Process data

        Args:
            data: list of dictionary

        Returns:
            dictionary of batch for each
        """

        batch = {
            "obs": {
                "image": np.concatenate([el["obs"]["image"] for el in data]) if "image" in data[0]["obs"] else None,
                "sensor": np.concatenate([el["obs"]["sensor"] for el in data]) if "sensor" in data[0]["obs"] else None
            },
            "act": np.concatenate([np.expand_dims(el["act"], axis=0) for el in data]),
            "next_obs": {
                "image": np.concatenate([el["next_obs"]["image"] for el in data]) if "image" in data[0]["obs"] else None,
                "sensor": np.concatenate([el["next_obs"]["sensor"] for el in data]) if "sensor" in data[0]["obs"] else None
            },
            "rew": np.expand_dims(np.concatenate([el["rew"] for el in data]), axis=-1),
            "done": np.expand_dims(np.concatenate([el["done"] for el in data]), axis=-1)
        }
        return batch

    def sample(self, size=256):
        idx = np.random.randint(0, len(self.data), size=size)
        tmp = np.array(self.data)
        return self.process(tmp[idx])


if __name__ == "__main__":
    from crazycar.environments import Environment
    from crazycar.agents import ImageAgent, SensorAgent, Racecar
    import math

    rb = Replay()
    env = Environment()
    agents = [Racecar, Racecar]
    positions = [[2.5, 6, math.pi * 2 / 2.0], [2.5, 4, math.pi * 2 / 2.0]]
    for agent, pos in zip(agents, positions):
        env.insert_car(agent, pos)

    obs = env.reset()

    while 1:
        act = np.array([[1, 0], [0, 0]])
        next_obs, rew, done, info = env.step(np.array([[1, 0], [0, 0]]))
        rb.store({
            "obs": obs[0],
            "act": act[0],
            "next_obs": next_obs[0],
            "rew": rew[0],
            "done": done,
        })
        break

    tmp = rb.sample()

    print(tmp['obs']['image'].shape)
    print(tmp['obs']['sensor'].shape)
    print(tmp['act'].shape)
    print(tmp['next_obs']['image'].shape)
    print(tmp['next_obs']['sensor'].shape)
    print(tmp['rew'].shape)
    print(tmp['done'].shape)


