import click

from pysim.environment import CrazyCar
from pysim.constants import *

def main():
    env = CrazyCar(renders=True, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP)

    # reset 
    state = env.reset()

    while True: 
        obs, reward, done, info = env.step([0, 0])
        print(obs)


if __name__ == '__main__':
	main()