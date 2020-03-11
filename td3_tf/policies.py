import tensorflow as tf
import tensorflow.keras as k


class AbstractModel(k.Model):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def soft_update(self, other_network, tau):
        other_variables = other_network.trainable_variables
        current_variables = self.trainable_variables

        for (current_var, other_var) in zip(current_variables, other_variables):
            current_var.assign((1. - tau) * current_var + tau * other_var)

    def hard_update(self, other_network):
        self.soft_update(other_network, tau=1.)


class Actor(AbstractModel):

    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)

        self.pi = k.Sequential([
            k.layers.Dense(units=256, input_shape=(None, obs_dim), activation=k.activations.relu),
            k.layers.Dense(units=256, activation=k.activations.relu),
            k.layers.Dense(units=act_dim, activation=k.activations.tanh)
        ])

        self.pi.build()

    def call(self, states):
        out = self.pi(states)
        return out


class Critic(AbstractModel):

    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)

        self.q1 = k.Sequential([
            k.layers.Dense(units=256, input_shape=(None, obs_dim + act_dim), activation=k.activations.relu),
            k.layers.Dense(units=256, activation=k.activations.relu),
            k.layers.Dense(units=1)
        ])

        self.q2 = k.Sequential([
            k.layers.Dense(units=256, input_shape=(None, obs_dim + act_dim), activation=k.activations.relu),
            k.layers.Dense(units=256, activation=k.activations.relu),
            k.layers.Dense(units=1)
        ])

        self.q1.build()
        self.q2.build()

    def call(self, obs, act):
        concat = tf.concat([obs, act], axis=1)
        return [self.q1(concat), self.q2(concat)]

    def q1_forward(self, obs, act):
        concat = tf.concat([obs, act], axis=1)
        return self.q1(concat)


class ActorCritic:

    def __init__(self, obs_dim, act_dim, actor_lr=1e-4, critic_lr=1e-4):
        super().__init__()

        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.hard_update(self.actor)
        self.actor.optimizer = k.optimizers.Adam(actor_lr)

        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.hard_update(self.critic)
        self.critic.optimizer = k.optimizers.Adam(critic_lr)

    @tf.function
    def act(self, obs):
        return self.actor(obs)
