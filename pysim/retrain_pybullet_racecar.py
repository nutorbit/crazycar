#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
#from pybullet_envs.bullet.racecarGymEnv import  RacecarGymEnv
#from pysim.crazycarGymEnv import CrazycarGymEnv
from . import CrazycarGymEnv

from baselines import deepq

import datetime



def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -10
    return is_solved


def main(modelfile="racecar_model.pkl"):
  
    env = CrazycarGymEnv(renders=False,isDiscrete=True)
    act = deepq.load(modelfile)
    model = act._act_params['q_func']
    dir(model)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,               
        max_timesteps=1*1000000, #2000000, #200000,  #60000  #200000
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback
    )
    print("Saving model to racecar_model.pkl")
    act.save("racecar_model_next.pkl")


if __name__ == '__main__':
    if(len(sys.argv) != 2):
        main()
    else:
        main(sys.argv[1])
