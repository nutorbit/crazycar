import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym

from . import CrazycarGymEnv

from baselines import deepq

def main():
    env = CrazycarGymEnv(renders=False,isDiscrete=True,calibration=True)
    obsStart, done = env.reset(), False
    carStart, carOri = env.getCarBasePositionAndOrientation()
    print("start: {}".format(carStart))
    while not done:
        env.render()
        obs, rew, done, _ = env.step(0)
        if env.getTime() > 3.5:
            done = True
    carEnd, carEndOri = env.getCarBasePositionAndOrientation()                
    print("end:  {}".format(carEnd))    
    diff = carEnd[1] - carStart[1]
    print(diff)
    print("expected v for speed=200\n ~0.93657,                 3.5s,                   ~3.263m,   SET 'speedMultiplier' in crazycar.py!")
    print("v={} m/s t={}s dist={}".format(diff/env.getTime(), env.getTime(), diff))
    
    
if __name__ == '__main__':
    main()
