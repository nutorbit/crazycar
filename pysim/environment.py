import math
from abc import ABC

import gym
import time
import pybullet
import random
import pybullet_data
import numpy as np

from collections import deque
from pybullet_envs.bullet import bullet_client
from gym import spaces
from pysim.constants import *

from pysim import track
from pysim import agent
from pysim import positions
from pysim.utils import get_reward_function


class CrazyCar(ABC):

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 renders=False,
                 origin=None,
                 reward_name="theta"):

        if origin is None:
            origin = [0, 0, 0]

        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._actionRepeat = ACTION_REP
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = DISCRETE_ACTION
        self._origin = origin
        self._collisionCounter = 0
        self._poscar = positions.CarPosition(origin)
        self._reward_function = get_reward_function()[reward_name]
        self._speed = 0

        if self._renders:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        obs = self.reset()

        # define observation space
        observationDim = obs.shape
        observation_high = np.full(observationDim, 1)
        observation_low = np.zeros(observationDim)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        # define action space
        if self._isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            action_low = np.array([0.5, -1])
            action_high = np.array([1, 1])

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

    def reset(self, newCarPos=None, random_position=True, PosIndex=1):

        # reset
        self._reset()

        # spawn car
        if newCarPos is None:
            if random_position:
                carPos = self._poscar.getNewPosition(random.randint(1, 11))
            else:
                carPos = self._poscar.getNewPosition(PosIndex)
        else:
            carPos = newCarPos

        self._racecar = agent.Racecar(self._p, self._origin, carPos, self._planeId, urdfRootPath=self._urdfRoot,
                                      timeStep=self._timeStep, direction_field=self._direction_field)

        # get observation
        self._observation = self._racecar.getObservation()

        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def getAction(self, action):

        realaction = action.ravel()

        self._speed = realaction[0]

        return realaction

    def step(self, action):

        realaction = self.getAction(action)

        self._racecar.applyAction(realaction)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()

        self._observation = self._racecar.getObservation()

        self._envStepCounter += 1

        reward = self._reward()
        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        return self._envStepCounter > MAX_STEPS or self._terminate or self._racecar.atGoal

    def _reward(self):

        diffAngle = self._racecar.diffAngle()
        # angleField = self._racecar.getAngleField()

        # cross road
        # isCross = angleField in list(range(45, 360, 90))

        # sensors
        sensors = self._racecar.getSensor()

        # calculate reward
        reward = eval(self._reward_function)

        # x, y, yaw = self._racecar.getCoordinate()

        if self._racecar.isCollision():
            reward = -50
            self._collisionCounter += 1

            self._terminate = True

        if math.pi - math.pi / 4 <= diffAngle <= math.pi + math.pi / 4:
            reward = -50
            # backward
            self._terminate = True

        return reward

    @property
    def p(self):
        return self._p


class SingleControl(CrazyCar):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        action_low = np.array([-1])
        action_high = np.array([1])

        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def getAction(self, action):
        realaction = [1, action]

        self._speed = realaction[0]

        return realaction


class MultiCar(CrazyCar):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atGoal = None

    def reset(self):

        # reset
        self._reset()

        self.atGoal = None

        carPos1 = [2.9 - 0.7 / 2, 1.5, math.pi / 2.0]
        carPos2 = [2.9 - 0.7 / 2, 1.2, math.pi / 2.0]

        carPoses = [carPos1, carPos2]
        np.random.shuffle(carPoses)

        # 2 agent
        self._racecars = [
            agent.Racecar(self._p, self._origin, carPos, self._planeId, urdfRootPath=self._urdfRoot, \
                          timeStep=self._timeStep, direction_field=self._direction_field)
            for carPos in carPoses
        ]

        for i in range(2):
            print(f"Agent {i} at position {carPoses[i][:-1]}")

        return self.getObservationAll()

    def getObservationAll(self):
        return np.array([racecar.getObservation() for racecar in self._racecars])

    def _termination(self):
        return self._envStepCounter > MAX_STEPS or self._terminate or \
               (self._racecars[0].atGoal or self._racecars[1].atGoal) or \
               (self._racecars[0].nCollision > 5 and self._racecars[1].nCollision > 5) or \
               self._racecars[0].nCollision > 100 or self._racecars[1].nCollision > 100

    def step(self, action):

        rewards = []
        obses = []

        for car_id, (racecar, realaction) in enumerate(zip(self._racecars, action)):

            racecar.applyAction(realaction)

            for i in range(self._actionRepeat):
                self._p.stepSimulation()

            obs = racecar.getObservation()
            reward = self._reward(racecar, realaction, obs)

            if racecar.atGoal:
                reward += 1000
                self.atGoal = car_id

            rewards.append(reward)
            obses.append(obs)

        self._envStepCounter += 1

        # observation = self.getObservationAll()

        done = self._termination()

        return np.array(obses), np.array(rewards), done, {}

    def _reward(self, racecar, realaction, obs):

        diffAngle = racecar.diffAngle()
        opponent = int(4 in np.unique(obs))
        speed = realaction[0]

        bonus = ((1 - opponent) * speed) + (opponent * speed * 0.1)

        reward = speed * math.cos(diffAngle) - speed * math.sin(diffAngle) + bonus

        if racecar.isCollision():
            reward -= 50
            racecar.nCollision += 1

        return reward

    def report(self):

        ncollisions = []

        for racecar in self._racecars:
            ncollisions.append(racecar.nCollision)

        return ncollisions, self.atGoal


class FrameStack:
    def __init__(self, env, k=4):
        self.k = k
        self.env = env
        self.frames = deque([], maxlen=k)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        shape = self.env.observation_space.shape
        shape = shape[:-1] + (self.k,)
        observation_high = np.full(shape, 1)
        observation_low = np.zeros(shape)
        return spaces.Box(observation_low, observation_high, dtype=np.float32)

    @property
    def _reward_function(self):
        return self.env._reward_function

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), rew, done, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)


if __name__ == '__main__':
    env = SingleControl(renders=True)
    env = FrameStack(env)
    # print(env.observation_space.shape)
    # env.reset([2.9 - 0.7/2, 0.8, math.pi/2.0])
    # env.reset(random_position=False, PosIndex=6)
    # env.p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[1.5, 3.3, 0])
    obs = env.reset(random_position=False)
    print(obs.shape)
    # x - [2.1, 2.9]
    while 1:
        # obs, rew, done, _ = env.step(0)
        # print(obs.shape)
        pass
