import math

from absl import app

from crazycar.environments import Environment
from crazycar.agents import ImageAgent


def rollout(env):
    """
    Rollout environment

    Args:
        env: environment object
    """



def main(_):
    env = Environment()
    agents = [ImageAgent, ImageAgent, ImageAgent]
    positions = [[2.5, 6, math.pi * 2 / 2.0], [2.5, 6, math.pi * 2 / 2.0]]
    for agent, pos in zip(agents, positions):
        env.insert_car(agent, pos)
    rollout(env)



if __name__ == "__main__":
    app.run(main)

