import tensorflow as tf
import numpy as np
import sonnet as snt

from crazycar.algos_tf.common import Replay
from crazycar.agents.constants import SENSOR_SHAPE, CAMERA_SHAPE


class BaseNetwork(snt.Module):
    """
    Base class for policy network and value network
    """

    def __init__(self, name=None):
        super().__init__(name=name)

    # @tf.function
    def initialize_input(self):
        tmp = {
            "sensor": tf.ones((1, ) + SENSOR_SHAPE),
            "image": tf.ones((1, ) + CAMERA_SHAPE)
        }

        # filter perception
        if self.enc.__class__.__name__ == "Sensor":
            tmp.pop("image", None)
        if self.enc.__class__.__name__ == "Image":
            tmp.pop("sensor", None)

        # filter for
        if self.__class__.__name__ == "Critic":
            _ = self(tmp, tf.ones((1, self.act_dim)))
        else:
            _ = self(tmp)

    # @tf.function
    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            # print(current_var.name, current_var.shape)
            # print(other_var.name, other_var.shape)
            current_var.assign((1. - tau) * current_var + tau * other_var)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)


class BaseModel(snt.Module):
    """
    Base class for Actor-Critic algorithm
    """

    def __init__(self, replay_size=int(1e5), name=None):
        super().__init__(name=name)
        self.rb = Replay(replay_size)

    def write_metric(self, metric, step):
        raise NotImplementedError

    def actor_loss(self, batch):
        raise NotImplementedError

    def critic_loss(self, batch):
        raise NotImplementedError

    @staticmethod
    def rescale(act, low=0, high=1):
        """
        Rescale action from tanh [-1, 1] to [low, high]
        """

        act = low + (0.5 * (act + 1.0) * (high - low))

        return act

    @tf.function
    def update_actor(self, batch):
        with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
            with tf.GradientTape() as tape:
                loss = self.actor_loss(batch)

            # Optimize the actor
            grads = tape.gradient(loss, self.actor.trainable_variables)
            self.actor_opt.apply(grads, self.actor.trainable_variables)

        return loss

    @tf.function
    def update_critic(self, batch):
        with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
            with tf.GradientTape() as tape:
                loss = self.critic_loss(batch)

            # Optimize the critic
            grads = tape.gradient(loss, self.critic.trainable_variables)
            self.critic_opt.apply(grads, self.critic.trainable_variables)

        return loss

    def update_params(self, i, batch_size=256):
        batch = self.rb.sample(batch_size)

        critic_loss = self.update_critic(batch)
        actor_loss = self.update_actor(batch)

        # update target network
        if i % self.interval_target == 0:
            self.actor_target.soft_update(self.actor, self.tau)
            self.critic_target.soft_update(self.critic, self.tau)

        return {
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy()
        }

    def random_action(self):
        act = np.random.uniform(-1, 1, size=(1, self.actor.act_dim))
        if self.actor.act_dim == 2:
            # apply rescale to speed
            act[0][0] = self.rescale(act[0][0])
        return act

    # @tf.function
    def predict(self, obs):
        act = self.actor(obs)
        act = act.numpy()
        if len(act[0]) == 2:
            # apply rescale to speed
            act[0][0] = self.rescale(act[0][0])
        return act
