import tensorflow as tf

from tensorflow.keras import activations, optimizers

from crazycar.utils import make_mlp
from crazycar.algos.base import BaseNetwork, BaseModel


class Actor(BaseNetwork):
    """
    Actor for DDPG

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = encoder()
        self.pi = make_mlp(sizes=[self.enc.out_size] + hiddens + [act_dim], activation=activations.relu)

    def call(self, obs):
        x = self.enc(obs)
        x = self.pi(x)
        return x


class Critic(BaseNetwork):
    """
    Double Q for DDPG

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim, hiddens=[256, 256]):
        super().__init__()
        self.enc = encoder()
        # print([self.enc.out_size + act_dim] + hiddens + [1])
        self.q1 = make_mlp(sizes=[self.enc.out_size + act_dim] + hiddens + [1], activation=activations.relu)
        self.q2 = make_mlp(sizes=[self.enc.out_size + act_dim] + hiddens + [1], activation=activations.relu)

    def call(self, obs, act):
        x = self.enc(obs)
        x = tf.concat([x, act], axis=1)
        return self.q1(x), self.q2(x)


class DDPG(BaseModel):
    """
    Deep Deterministic Policy Gradient

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        lr: learning rate
        gamma: discount factor
        interval_target: number of iteration for update target network
        tau: polyak average
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim,
                 lr=1e-4,
                 gamma=0.9,
                 interval_target=2,
                 tau=0.05,
                 hiddens=[256, 256]):

        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        # define actor
        self.actor = Actor(encoder, act_dim, hiddens)
        self.actor_target = Actor(encoder, act_dim, hiddens)
        self.actor_target.hard_update(self.actor)

        # define critic
        self.critic = Critic(encoder, act_dim, hiddens)
        self.critic_target = Critic(encoder, act_dim, hiddens)
        self.critic_target.hard_update(self.critic)

        # define optimizer
        self.actor_opt = optimizers.Adam(lr=lr)
        self.critic_opt = optimizers.Adam(lr=lr)

    def actor_loss(self, batch):
        """
        L(s) = -E[Q(s, a)| a~u(s)]
        """

        act = self.actor(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        loss = -q1
        return loss

    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2

        Where,
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s')
        """

        next_act = self.actor_target(batch['next_obs'])
        q_target1, q_target2 = self.critic_target(batch['next_obs'], next_act)
        q_target = tf.minimum(q_target1, q_target2)
        y = batch['rew'] + (1 - batch['done']) * self.gamma * tf.stop_gradient(q_target)

        q1, q2 = self.critic(batch['obs'], batch['act'])

        loss1 = tf.reduce_mean(tf.square(y - q1))
        loss2 = tf.reduce_mean(tf.square(y - q2))

        return loss1 + loss2


if __name__ == "__main__":
    from crazycar.encoder import Sensor, Image
    from crazycar.utils import set_seed

    from crazycar.agents.constants import DISTANCE_SENSORS, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_DEPT

    set_seed()
    agent = DDPG(Sensor, 5)

    tmp = {
        "sensor": tf.ones((1, len(DISTANCE_SENSORS))),
        "image": tf.ones((1, CAMERA_HEIGHT, CAMERA_WIDTH, CAMERA_DEPT))
    }

    print(agent.predict(tmp))
