import tensorflow as tf
import sonnet as snt

from tensorflow.keras import layers, activations


class Sensor(snt.Module):
    """
    For sensor feature with 256 feature
    """

    def __init__(self, name="sensor"):
        super().__init__(name=name)
        self.encode = snt.Sequential([
            layers.Dense(256, activation=activations.tanh)
        ])

    @tf.function
    def __call__(self, obs):
        x = self.encode(obs['sensor'])
        return x


class Image(snt.Module):
    """
    For image feature with 256 feature
    """

    def __init__(self, name="image"):
        super().__init__(name=name)
        self.encode = ImpalaCNN()

    @tf.function
    def __call__(self, obs):
        x = self.encode(obs['image'])
        return x


class Combine(snt.Module):
    """
    Use all feature with 512 feature
    """

    def __init__(self, name="combine"):
        super().__init__(name=name)
        self.image_reps = ImpalaCNN()
        self.sensor_reps = snt.Sequential([
            layers.Dense(256, activation=activations.tanh),
        ])

    @tf.function
    def __call__(self, obs):
        image_reps = self.image_reps(obs['image'])
        sensor_reps = self.sensor_reps(obs['sensor'])
        concat = tf.concat([image_reps, sensor_reps], axis=1)
        return concat


class ImpalaCNN(snt.Module):
    """
    The CNN architecture used in the IMPALA paper.
    Ref: https://arxiv.org/abs/1802.01561
    """

    def __init__(self, name="impala_cnn"):
        super().__init__(name=name)
        l = []
        for depth_out in [16, 32, 32]:
            l.extend([
                layers.Conv2D(filters=depth_out, kernel_size=3, padding='same'),
                layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
        self.conv_layers = snt.Sequential(l)
        self.out = layers.Dense(256)

    @tf.function
    def __call__(self, x):
        x = self.conv_layers(x)
        x = tf.nn.relu(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.out(x)
        x = tf.nn.relu(x)
        return x


class ImpalaResidual(snt.Module):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, dept, name="impala_residual"):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(filters=dept, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2D(filters=dept, kernel_size=3, padding='same')

    @tf.function
    def __call__(self, x):
        out = tf.nn.relu(x)
        out = self.conv1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        return out + x


if __name__ == "__main__":
    tmp = tf.ones((5, 10))
    I = Combine()
    print(I({"sensor": tf.ones((256, 10)), "image": tf.ones((256, 20, 20, 5))}))
