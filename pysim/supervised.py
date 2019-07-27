from stable_baselines.gail import ExpertDataset
from stable_baselines.gail import generate_expert_traj
from . import CrazycarGymEnv3

from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.ppo2 import PPO2
from stable_baselines import A2C, ACER


# env = CrazycarGymEnv3(renders=True, isDiscrete=True, actionRepeat=2)

def expert(_obs):

    x = -1
    while not (0 <= x <= 5):
        x = int(input())

    fwd = [1, 1, 1, 1, 1, 1] # 0 -> 5
    steerings = [60, 75, 105, 106, 136, 151]

    return x

def main():

    # supervised
    # generate_expert_traj(expert, 'supervised', env, n_episodes=1)

    # training
    # dataset = ExpertDataset('supervised.npz')

    env = CrazycarGymEnv3(renders=False, isDiscrete=False, actionRepeat=2)
    env = SubprocVecEnv([lambda: env for _ in range(4)])
    # env = VecNormalize(env, norm_reward=False)
    # print(env.action_space.high)
    # print(env.action_space.low)
    policy = MlpPolicy
    policy_kwargs = dict(net_arch=[32, 32, dict(vf=[32], pi=[32])])

    model = PPO2(
        policy,
        env,
        n_steps=1024,
        nminibatches=32,
        lam=0.95,
        gamma=0.8,
        noptepochs=10,
        vf_coef=1,
        ent_coef=0.01,
        cliprange=0.2,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    # model.pretrain(dataset, 1000)

    model.learn(total_timesteps=100000)

    env = CrazycarGymEnv3(renders=True, isDiscrete=False, actionRepeat=2)

    while True:
        state = env.reset()
        done = False
        i_step = 0
        ep_reward = 0
        while not done:
            action = model.predict(state)[0]
            # print(action)
            # action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(action, reward)
            state = next_state
            i_step += 1
            ep_reward += reward
        print(ep_reward)


if __name__ == '__main__':
    main()