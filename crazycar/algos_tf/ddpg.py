import tensorflow as tf
import sonnet as snt

from crazycar.algos_tf.common import make_mlp
from crazycar.algos_tf.base import BaseNetwork, BaseModel


class Actor(BaseNetwork):
    """
    Actor for DDPG

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim, hiddens=[256, 256], name=None):
        super().__init__(name=name)
        self.enc = encoder()
        self.act_dim = act_dim
        self.pi = make_mlp(
            sizes=hiddens + [act_dim],
            activation=tf.nn.tanh,
            output_activation=tf.nn.tanh
        )
        self.initialize_input()

    @tf.function
    def __call__(self, obs):
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

    def __init__(self, encoder, act_dim, hiddens=[256, 256], name=None):
        super().__init__(name=name)
        self.enc = encoder()
        self.act_dim = act_dim
        # print([self.enc.out_size + act_dim] + hiddens + [1])
        self.q1 = make_mlp(sizes=hiddens + [1], activation=tf.nn.tanh)
        self.q2 = make_mlp(sizes=hiddens + [1], activation=tf.nn.tanh)
        self.initialize_input()

    @tf.function
    def __call__(self, obs, act):
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
        replay_size: buffer size
        hiddens: NO. units for each layers
    """

    def __init__(self, encoder, act_dim,
                 lr=3e-4,
                 gamma=0.9,
                 interval_target=2,
                 tau=0.05,
                 replay_size=int(1e6),
                 hiddens=[256, 256],
                 name="DDPG"):

        super().__init__(replay_size, name=name)

        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        # define actor
        self.actor = Actor(encoder, act_dim, hiddens, name="actor")
        self.actor_target = Actor(encoder, act_dim, hiddens, name="actor_target")
        self.actor_target.hard_update(self.actor)

        # define critic
        self.critic = Critic(encoder, act_dim, hiddens, name="critic")
        self.critic_target = Critic(encoder, act_dim, hiddens, name="critic_target")
        self.critic_target.hard_update(self.critic)

        # define optimizer
        self.actor_opt = snt.optimizers.Adam(lr, name="actor_optimizer")
        self.critic_opt = snt.optimizers.Adam(lr, name="critic_optimizer")

    @tf.function
    def actor_loss(self, batch):
        """
        L(s) = -E[Q(s, a)| a~u(s)]
        """

        act = self.actor(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        loss = -tf.reduce_mean(q1)
        return loss

    @tf.function
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

    def write_metric(self, metric, step):
        tf.summary.scalar("loss/actor_loss", metric['actor_loss'], step)
        tf.summary.scalar("loss/critic_loss", metric['critic_loss'], step)


if __name__ == "__main__":
    from crazycar.algos_tf.encoder import Combine
    from crazycar.utils import set_seed

    from crazycar.agents.constants import SENSOR_SHAPE, CAMERA_SHAPE

    set_seed()

    tmp = {
        "sensor": tf.ones((1, ) + SENSOR_SHAPE),
        "image": tf.ones((1, ) + CAMERA_SHAPE)
    }

    agent = DDPG(Combine, 2)
    #

    #
    # agent.actor_target.hard_update(agent.actor)
    # p1 = agent.critic(tmp, np.zeros((1, 2), dtype='float32'))
    # p2 = agent.critic_target(tmp, np.zeros((1, 2), dtype='float32'))
    # print(p1)
    # print(p2)

    print(len(agent.actor.trainable_variables))
    print(len(agent.actor_target.trainable_variables))

    # for l, r in zip(agent.actor.enc.image_reps.trainable_variables, agent.actor_target.enc.image_reps.trainable_variables):
    #     # r.assign(l)
    #     print(l.name, r.name, np.mean(l == r))
    # agent.actor.summary()

    # print(agent.predict(tmp))
