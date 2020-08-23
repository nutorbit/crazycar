import tensorflow as tf

from tensorflow.keras import layers, activations

from crazycar.agents.constants import DISTANCE_SENSORS, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_DEPT


class Sensor(tf.keras.Model):
    """
    For sensor feature with 256 feature
    """

    def __init__(self):
        super().__init__()
        self.encode = tf.keras.Sequential([
            layers.Input(len(DISTANCE_SENSORS) + 2),
            layers.Dense(256, activation=activations.tanh)
        ])
        self.out_size = 256

    def call(self, obs):
        x = self.encode(obs['sensor'])
        return x


class Image(tf.keras.Model):
    """
    For image feature with 256 feature
    """

    def __init__(self):
        super().__init__()
        self.encode = ImpalaCNN()
        self.out_size = 256

    def call(self, obs):
        x = self.encode(obs['image'])
        return x


class Combine(tf.keras.Model):
    """
    Use all feature with 512 feature
    """

    def __init__(self):
        super().__init__()
        self.image_reps = ImpalaCNN()
        self.sensor_reps = tf.keras.Sequential([
            layers.Input(len(DISTANCE_SENSORS)),
            layers.Dense(256, activation=activations.tanh)
        ])
        self.out_size = 512

    def call(self, obs):
        image_reps = self.image_reps(obs['image'])
        sensor_reps = self.sensor_reps(obs['sensor'])
        concat = tf.concat([image_reps, sensor_reps], axis=1)
        return concat


class ImpalaCNN(tf.keras.Model):
    """
    The CNN architecture used in the IMPALA paper.
    Ref: https://arxiv.org/abs/1802.01561
    """

    def __init__(self):
        super().__init__()
        l = [layers.Input((CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_DEPT))]
        for depth_out in [16, 32, 32]:
            l.extend([
                layers.Conv2D(filters=depth_out, kernel_size=3, padding='same'),
                layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
                ImpalaResidual(depth_out),
                ImpalaResidual(depth_out),
            ])
        self.conv_layers = tf.keras.Sequential(l)
        self.out = layers.Dense(256)

    def call(self, x):
        x = self.conv_layers(x)
        x = tf.nn.relu(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.out(x)
        x = tf.nn.relu(x)
        return x


class ImpalaResidual(tf.keras.Model):
    """
    A residual block for an IMPALA CNN.
    """

    def __init__(self, dept):
        super().__init__()
        self.conv1 = layers.Conv2D(filters=dept, kernel_size=3, padding='same')
        self.conv2 = layers.Conv2D(filters=dept, kernel_size=3, padding='same')

    def call(self, x):
        out = tf.nn.relu(x)
        out = self.conv1(out)
        out = tf.nn.relu(out)
        out = self.conv2(out)
        return out + x


if __name__ == "__main__":
    tmp = tf.ones((5, 10))
    I = Combine()
    print(I({"sensor": tf.ones((5, 10)), "image": tf.ones((5, 10, 10, 5))}))
