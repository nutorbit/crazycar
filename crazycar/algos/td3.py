import tensorflow as tf

from crazycar.algos.ddpg import DDPG


class TD3(DDPG):
    """
    Twin Delayed Deep Deterministic Policy Gradient

    Args:
        encoder: class from crazycar.encoder
        act_dim: number of action
        lr: learning rate
        gamma: discount factor
        interval_target: number of iteration for update target network
        tau: polyak average
        replay_size: buffer size
        hiddens: NO. units for each layers
        target_noise: noise in target network
        noise_clip: noise clip
    """

    def __init__(self, encoder,
                 act_dim,
                 lr=1e-4,
                 gamma=0.9,
                 interval_target=2,
                 tau=0.05,
                 replay_size=int(1e5),
                 hiddens=[256, 256],
                 target_noise=0.2,
                 noise_clip=0.5,
                 name="TD3"):

        super().__init__(encoder, act_dim, lr, gamma, interval_target, tau, replay_size, hiddens, name=name)
        self.target_noise = target_noise
        self.noise_clip = noise_clip

    @tf.function
    def critic_loss(self, batch):
        """
        L(s, a) = (y - Q(s,a))^2

        Where,
            y(s, a) = r(s, a) + (1 - done) * gamma * Q'(s', a'); a' ~ u'(s') + noise
        """

        # next action
        print(batch['next_obs'])
        next_act = self.actor_target(batch['next_obs'])
        noise = self.get_noise(shape=next_act.shape)
        next_act = tf.clip_by_value(next_act + noise, -1, 1)

        q_target1, q_target2 = self.critic_target(batch['next_obs'], next_act)
        q_target = tf.minimum(q_target1, q_target2)
        y = batch['rew'] + (1 - batch['done']) * self.gamma * tf.stop_gradient(q_target)

        q1, q2 = self.critic(batch['obs'], batch['act'])

        loss1 = tf.reduce_mean(tf.square(y - q1))
        loss2 = tf.reduce_mean(tf.square(y - q2))

        return loss1 + loss2

    def get_noise(self, shape):
        """
        Random noise for exploration and prevent overestimate in Q

        Args:
            shape: shape of noise
        """

        noise = tf.random.normal(shape=shape)
        noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
        return noise

    def predict(self, obs, test=False):
        act = self.actor(obs)
        if not test:
            noise = self.get_noise(shape=act.shape)
            act = tf.clip_by_value(act + noise, -1, 1)
        return act.numpy()


if __name__ == "__main__":
    from crazycar.encoder import Sensor, Image
    from crazycar.utils import set_seed

    from crazycar.agents.constants import SENSOR_SHAPE, CAMERA_SHAPE

    set_seed()
    agent = TD3(Sensor, 5)

    tmp = {
        "sensor": tf.ones((1, ) + SENSOR_SHAPE),
        "image": tf.ones((1, ) + CAMERA_SHAPE)
    }

    print(agent.predict(tmp, test=True))
