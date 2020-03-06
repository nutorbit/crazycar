import gym

from td3_tf.td3 import TD3

if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")

    model = TD3(env)
    model.learn(10000)