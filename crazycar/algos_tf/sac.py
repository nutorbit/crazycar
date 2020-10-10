import tensorflow as tf
import sonnet as snt

from tensorflow_probability import distributions

from crazycar.algos_tf.common import make_mlp
from crazycar.algos_tf.base import BaseModel, BaseNetwork


EPS = 1e-16


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
        self.hidden = make_mlp(
            sizes=hiddens,
            activation=tf.nn.relu
        )
        self.mean = snt.Linear(act_dim)
        self.log_std = snt.Linear(act_dim)
        self.initialize_input()

    @tf.function
    def __call__(self, obs):
        x = self.enc(obs)
        x = self.hidden(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        return mean, log_std

    @tf.function
    def sample(self, obs):
        mean, log_std = self(obs)
        std = tf.exp(log_std)
        dist = distributions.Normal(mean, std)

        # sample the action and apply tanh squashing
        action_sample = dist.sample()
        action = tf.tanh(action_sample)

        # calculate log prob
        log_prob = dist.log_prob(action_sample)
        log_prob -= tf.reduce_sum(tf.math.log(1 - action**2 + EPS), axis=1, keepdims=True)

        return action, log_prob


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
        self.q1 = make_mlp(sizes=hiddens + [1], activation=tf.nn.relu)
        self.q2 = make_mlp(sizes=hiddens + [1], activation=tf.nn.relu)
        self.initialize_input()

    @tf.function
    def __call__(self, obs, act):
        x = self.enc(obs)
        x = tf.concat([x, act], axis=1)
        return self.q1(x), self.q2(x)


class CriticV(Critic):
    def __init__(self, encoder, act_dim, hiddens=[256, 256], name=None):
        super().__init__(encoder, act_dim, hiddens, name=name)

    @tf.function
    def __call__(self, obs):
        x = self.enc(obs)
        # print(x)
        return self.q1(x), self.q2(x)


class SAC(BaseModel):
    """
    Soft Actor-Critic

    Args:
        ...
    """

    def __init__(self, encoder, act_dim,
                 lr=3e-4,
                 gamma=0.99,
                 interval_target=2,
                 tau=0.05,
                 replay_size=int(1e6),
                 hiddens=[256, 256],
                 name="SAC"):

        super().__init__(replay_size, name=name)

        self.tau = tau
        self.gamma = gamma
        self.interval_target = interval_target

        # define actor
        self.actor = Actor(encoder, act_dim, hiddens, name="actor")

        # define critic
        self.critic = Critic(encoder, act_dim, hiddens, name="critic")

        # define critic v
        self.critic_v = CriticV(encoder, act_dim, hiddens, name="critic_v")
        self.critic_v_target = CriticV(encoder, act_dim, hiddens, name="critic_v_target")
        self.critic_v_target.hard_update(self.critic_v)

        # define alpha
        self.log_alpha = tf.Variable(.0, dtype=tf.float32, name="log_alpha", trainable=True)
        self.target_entropy = -tf.constant(act_dim, dtype=tf.float32)

        # define optimizer
        self.actor_opt = snt.optimizers.Adam(lr, name="actor_optimizer")
        self.critic_opt = snt.optimizers.Adam(lr, name="critic_optimizer")
        self.critic_v_opt = snt.optimizers.Adam(lr, name="critic_v_optimizer")
        self.alpha_opt = snt.optimizers.Adam(lr, name="alpha_optimizer")

    @tf.function
    def actor_loss(self, batch):
        """
        L(s) = -E[Q(s, a)| a~u(s)]

        Where,
            Q is a soft-Q: Q - alpha * log_prob
        """

        act, log_prob = self.actor.sample(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        min_q = tf.minimum(q1, q2)
        loss = tf.reduce_mean(tf.exp(self.log_alpha) * log_prob - min_q)

        return loss

    @tf.function
    def critic_v_loss(self, batch):
        """
        ...
        """

        v1, v2 = self.critic_v(batch['obs'])
        v = tf.minimum(v1, v2)

        act, log_prob = self.actor.sample(batch['obs'])
        q1, q2 = self.critic(batch['obs'], act)
        min_q = tf.minimum(q1, q2)

        target_v = tf.stop_gradient(min_q - tf.exp(self.log_alpha) * log_prob)
        td_v = tf.reduce_mean((target_v - v) ** 2)
        # print(batch)

        return td_v

    @tf.function
    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2

        Where,
            Q is a soft-Q: Q - alpha * log_prob
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s')
        """

        q1, q2 = self.critic(batch['obs'], batch['act'])
        next_v_target1, next_v_target2 = self.critic_v(batch['next_obs'])
        next_v_target = tf.minimum(next_v_target1, next_v_target2)

        target_q = tf.stop_gradient(
            batch['rew'] + (1 - batch['done']) * self.gamma * next_v_target
        )

        td_q1 = tf.reduce_mean((target_q - q1) ** 2)
        td_q2 = tf.reduce_mean((target_q - q2) ** 2)

        return td_q1 + td_q2

    @tf.function
    def alpha_loss(self, batch):
        """
        L = -(alpha * log_prob + target_entropy)
        """

        act, log_prob = self.actor.sample(batch['obs'])
        # print(act, log_prob)
        loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_prob + self.target_entropy))
        return loss

    @tf.function
    def update_critic_v(self, batch):
        with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
            with tf.GradientTape() as tape:
                loss = self.critic_v_loss(batch)

            # Optimize the critiv v
            grads = tape.gradient(loss, self.critic_v.trainable_variables)
            self.critic_v_opt.apply(grads, self.critic_v.trainable_variables)

        return loss

    @tf.function
    def update_alpha(self, batch):
        with tf.device('/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'):
            with tf.GradientTape() as tape:
                loss = self.alpha_loss(batch)

            # Optimize the alpha
            grads = tape.gradient(loss, [self.log_alpha])
            self.alpha_opt.apply(grads, [self.log_alpha])

        return loss

    def update_params(self, i, batch_size=256):
        batch = self.rb.sample(batch_size)

        critic_loss = self.update_critic(batch)
        critic_v_loss = self.update_critic_v(batch)
        actor_loss = self.update_actor(batch)
        alpha_loss = self.update_alpha(batch)

        # update target network
        if i % self.interval_target == 0:
            self.critic_v_target.soft_update(self.critic_v, self.tau)

        return {
            "actor_loss": actor_loss.numpy(),
            "critic_loss": critic_loss.numpy(),
            "critic_v_loss": critic_v_loss.numpy(),
            "alpha_loss": alpha_loss.numpy(),
            "alpha": tf.exp(self.log_alpha).numpy()
        }

    def write_metric(self, metric, step):
        tf.summary.scalar("loss/actor_loss", metric['actor_loss'], step)
        tf.summary.scalar("loss/critic_loss", metric['critic_loss'], step)
        tf.summary.scalar("loss/critic_v_loss", metric['critic_v_loss'], step)
        tf.summary.scalar("loss/alpha_loss", metric['alpha_loss'], step)
        tf.summary.scalar("track/alpha", metric['alpha'], step)

    # @tf.function
    def predict(self, obs):
        act, _ = self.actor.sample(obs)
        act = act.numpy()
        # if len(act[0]) == 2:
        #     # apply rescale to speed
        #     act[0][0] = self.rescale(act[0][0])
        return act


if __name__ == "__main__":
    from crazycar.algos_tf.encoder import Sensor
    from crazycar.utils import set_seed

    from crazycar.agents.constants import SENSOR_SHAPE, CAMERA_SHAPE

    set_seed()
    agent = SAC(Sensor, 5)

    tmp = {
        "sensor": tf.ones((1, ) + SENSOR_SHAPE),
        "image": tf.ones((1, ) + CAMERA_SHAPE)
    }

    print(agent.predict(tmp))
