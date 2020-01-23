import math
import gym
import time
import pybullet
import random
import pybullet_data
import numpy as np

from pybullet_envs.bullet import bullet_client
from gym import spaces
from gym.utils import seeding
from pysim.constants import *

from pysim import track
from pysim import agent
from pysim import positions


class CrazyCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 500
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 renders=False,
                 origin=[0, 0, 0]):

        self._timeStep = 0.005
        self._urdfRoot = urdfRoot
        self._actionRepeat = ACTION_REP
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = DISCRETE_ACTION
        self._origin = origin
        self._collisionCounter = 0
        self._poscar = positions.CarPosition(origin)
        self._speed = 0

        if self._renders:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.seed()

        obs = self.reset()
        # print(obs.shape)

        # define observation space
        observationDim = obs.shape
        # print(observationDim)
        observation_high = np.full(observationDim, 10)
        observation_low = np.zeros(observationDim)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
        # print(self.observation_space)

        # define action space
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(9) #91
        else:
            action_low  = np.array([ -1])
            action_high = np.array([  1])

            self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def _reset(self):

        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)

        # time
        self._lastTime = time.time()

        # spawn plane
        self._planeId = self._p.loadURDF("./pysim/data/plane.urdf")

        # spawn race track
        self._direction_field = track.createRaceTrack(self._p, self._origin)

        # reset common variables
        for i in range(100):
            self._p.stepSimulation()

        self._p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        self._terminate = False
        self._collisionCounter = 0

    def reset(self, newCarPos=None):
        
        # reset
        self._reset()

        # spawn car
        if newCarPos is None:
            if RANDOM_POSITION:
                carPos = self._poscar.getNewPosition(random.randint(1, 11))
            else:
                carPos = self._poscar.getNewPosition(1)
        else:
            carPos = newCarPos

        self._racecar = agent.Racecar(self._p, self._origin, carPos, self._planeId, urdfRootPath=self._urdfRoot,
                                         timeStep=self._timeStep, direction_field=self._direction_field)
        
        # get observation
        self._observation = self._racecar.getObservation()

        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getAction(self, action):
        realaction = None

        if self._isDiscrete: # discrete action
            fwd = [1, 1, 1, 1, 1, 1, 1, 1, 1]
            steerings = [-1, -0.75, -0.45, -0.25, 0.00, 0.25, 0.45, 0.75, 1]
            forward = fwd[action]
            steer = steerings[action]
            realaction = [forward, steer]
            
        else: # continuous action
            # realaction = action
            # realaction = [action[0] + 1, action[1]]
            realaction = [1, action]
            # self.speed = realaction[0]

        self.speed = realaction[0]

        return realaction

    def step(self, action):
        if (self._renders):

            now_time = time.time()
            # if now_time-self._lastTime>.3:
            #     _ = self._racecar.getCameraImage()

        realaction = self.getAction(action)
        # print(self._racecar.diffAngle())

        self._racecar.applyAction(realaction)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self._racecar.getObservation()

            if self._termination():
                break
            self._envStepCounter += 1

        reward = self._reward()
        done   = self._termination()

        # if done:
        #     reward = -self._envStepCounter * 1e-3

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        return self._envStepCounter > MAX_STEPS or self._terminate

    def _reward(self):

        reward = self.speed * math.cos(self._racecar.diffAngle()) - self.speed * math.sin(self._racecar.diffAngle())
        # reward = 0

        # reward += -abs(self._observation[0] - self._observation[-2])*1e-2

        x, y, yaw = self._racecar.getCoordinate()

        if self._racecar.isCollision():
            # reward = -100
            self._collisionCounter += 1
            self._terminate = True
            
        if self._collisionCounter >= 10:
            self._terminate = True

        return reward


class MultiCar(CrazyCar):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self):

        # reset
        self._reset()

        carPos1 = [2.9 - 0.7/2, 0.8, math.pi/2.0]
        carPos2 = [2.9 - 0.7/2, 0.5, math.pi/2.0]

        carPoses = [carPos1, carPos2]

        # 2 agent
        self._racecars = [ 
            agent.Racecar(self._p, self._origin, carPos, self._planeId, urdfRootPath=self._urdfRoot,\
                             timeStep=self._timeStep, direction_field=self._direction_field)
            for carPos in carPoses 
        ]

        return self.getObservationAll()

    def getObservationAll(self):
        return np.array([racecar.getObservation() for racecar in self._racecars])

    def step(self, action):

        for racecar, realaction in zip(self._racecars, action):

            racecar.applyAction(realaction)

            for i in range(self._actionRepeat):
                self._p.stepSimulation()

                if self._termination():
                    break
                self._envStepCounter += 1


            reward = self._reward()
            done   = self._termination()

        observation = self.getObservationAll()

        return observation, reward, done, {}

    def _carCollision(self, carLinkIndex):
        res = []
        for racecar in self._racecars:
            res.append(racecar.isCollision())
        return res

    def _reward(self):
        pass