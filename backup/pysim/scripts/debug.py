from pysim.environment import CrazyCar


def main():
    env = CrazyCar(renders=True)
    # env = MultiCar(renders=True)

    # reset 
    obs = env.reset()

    while True: 
        # obs, reward, done, info = env.step([[0.2, 0], [0.11, 0]])
        obs, reward, done, info = env.step(0)
        print('------------------')
        print(obs)
        print(reward, done)
        print('------------------')


if __name__ == '__main__':
	main()