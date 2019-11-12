import click

from pysim import CrazycarGymEnv4
from pysim.constants import *

def main():
    env = CrazycarGymEnv4(renders=True, isDiscrete=DISCRETE_ACTION, actionRepeat=ACTION_REP)

    # reset 
    state = env.reset()

    print(state)

    while True: 
        pass


if __name__ == '__main__':
	main()