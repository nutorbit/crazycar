import random
import tensorflow as tf
import numpy as np

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

