import click

from pysim.environment import CrazyCar, MultiCar
from pysim.constants import *

def main():
    env = CrazyCar(renders=True)
    # env = MultiCar(renders=True)

    # reset 
    obs = env.reset()

    while True: 
        # obs, reward, done, info = env.step([[0.2, 0], [0.11, 0]])
        obs, reward, done, info = env.step([2, 0])
        print(obs)


if __name__ == '__main__':
	main()