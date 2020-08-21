import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

    # @tf.function
    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            current_var.assign((1. - tau) * current_var + tau * other_var)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)
