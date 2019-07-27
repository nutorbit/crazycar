from . import CrazycarGymEnv
import sys
import numpy as np
from baselines import deepq
from py4j.java_gateway import JavaGateway


def main(modelfile="racecar_model.pkl"):
    gateway = JavaGateway()
    posdata = []
    replay = False
    x, y, direction = 2.9 - 0.35, 4.4, np.math.pi / 2.0 #- 0.7854
    try:
        car = gateway.entry_point.getCarWithSimpleLocationEstimator(x*1000, y*1000, direction)
        estimator = car.getLocationEstimator()
        car.connect()
        env = CrazycarGymEnv(renders=False, isDiscrete=True)
        env.setRealCar(car)
        print(modelfile)
        act = deepq.load(modelfile)
        obs, done = env.reset(newCarPos=[x, y, direction]), False

        print("===================================")
        print("obs")
        print(obs)
        episode_rew = 0
        while not done:
            env.render()
            action = act(obs[None])[0]
            obs, rew, done, _ = env.step(action)
            car.setSpeed(200)
            car.waitForNextData()
            locations = estimator.getEstimatedLocations()
            loc = locations[0]
            posdata.append([loc.getX()/1000, loc.getY()/1000, loc.getDirection()])
            env.resetCarPositionAndOrientation(loc.getX()/1000, loc.getY()/1000, loc.getDirection())
            #car.setSteering(steering.astype(np.int32).item())
            episode_rew += rew
        print("Episode reward", episode_rew)
        car.setSpeed(0)
    finally:
        car.disconnect()
        gateway.close()
    if replay:
        input("Press Enter to continue...")
        env = CrazycarGymEnv(renders=True, isDiscrete=True)
        obs, done = env.reset(newCarPos=[x, y, direction]), False
        print("===================================")
        print("obs")
        print(obs)
        episode_rew = 0
        for location in posdata:
            env.render()
            action = act(obs[None])[0]
            obs, rew, done, _ = env.step(action)
            env.resetCarPositionAndOrientation(location[0], location[1], location[2])
            episode_rew += rew
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        main()
    else:
        main(sys.argv[1])
