import random
import tensorflow as tf
import sonnet as snt
import numpy as np

from collections import deque


def initial(seed=100):
    """
    Initial seed & gpu

    Args:
        seed: seed number
    """

    set_seed(seed)

    if tf.test.is_gpu_available():  # gpu limit
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

    l = []
    for i in range(1, len(sizes)):
        if i != len(sizes) - 1:
            l.extend([snt.Linear(sizes[i]), activation])
            # l.append(layers.Dense(sizes[i], activation=activation))
        else:
            l.append(snt.Linear(sizes[i]))
            if output_activation:
                l.append(output_activation)
            # l.append(layers.Dense(sizes[i], activation=output_activation))
    # return tf.keras.Sequential(l)
    # print(l)
    return snt.Sequential(l)


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
                "image": tf.convert_to_tensor(
                    np.concatenate([el["obs"]["image"] for el in data]), dtype=tf.float32
                ) if "image" in data[0]["obs"] else None,
                "sensor": tf.convert_to_tensor(
                    np.concatenate([el["obs"]["sensor"] for el in data]), dtype=tf.float32
                ) if "sensor" in data[0]["obs"] else None
            },
            "act": tf.convert_to_tensor(
                np.concatenate([np.expand_dims(el["act"], axis=0) for el in data]), dtype=tf.float32
            ),
            "next_obs": {
                "image": tf.convert_to_tensor(
                    np.concatenate([el["next_obs"]["image"] for el in data]), dtype=tf.float32
                ) if "image" in data[0]["obs"] else None,
                "sensor": tf.convert_to_tensor(
                    np.concatenate([el["next_obs"]["sensor"] for el in data]), dtype=tf.float32
                ) if "sensor" in data[0]["obs"] else None
            },
            "rew": tf.convert_to_tensor(
                np.expand_dims(np.concatenate([el["rew"] for el in data]), axis=-1), dtype=tf.float32
            ),
            "done": tf.convert_to_tensor(
                np.expand_dims(np.concatenate([el["done"] for el in data]), axis=-1), dtype=tf.float32
            )
        }

        # clear unwanted key
        if batch["obs"]["image"] is None:
            batch["obs"].pop("image")
        if batch["obs"]["sensor"] is None:
            batch["obs"].pop("sensor")

        if batch["next_obs"]["image"] is None:
            batch["next_obs"].pop("image")
        if batch["next_obs"]["sensor"] is None:
            batch["next_obs"].pop("sensor")

        return batch

    def sample(self, size=256):
        idx = list(range(len(self.data)))
        np.random.shuffle(idx)
        idx = idx[:size]
        tmp = np.array(self.data)
        return self.process(tmp[idx])