#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
#from pybullet_envs.bullet.racecarGymEnv import  RacecarGymEnv
#from pysim.crazycarGymEnv import CrazycarGymEnv
from . import CrazycarGymEnv
from . import CrazycarGymEnv3

from baselines import deepq
from . import mydeepq

import datetime



def callback(lcl, glb):
    # stop training if reward exceeds 199
    total = sum(lcl['episode_rewards'][-101:-1]) / 100
    totalt = lcl['t']
    is_solved = totalt > 2000 and total >= -10
#    print("totalt={}, total={}".format(totalt, total))
    return is_solved


def main():
  
    env = CrazycarGymEnv3(renders=False, isDiscrete=True, actionRandomized=False)
    model = deepq.models.mlp([32], layer_norm=True)
    #act = mydeepq.learn(
    act = deepq.learn(
            env,
        q_func=model,
        lr=1e-2,
        max_timesteps=80000, #10000, #2000000, #200000,  #60000  #200000
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        param_noise=True
#        old_state="racecar_model.pkl"
    )

    print("Saving model to racecar_model.pkl")
    act.save("racecar_model_test.pkl")


if __name__ == '__main__':
    main()
