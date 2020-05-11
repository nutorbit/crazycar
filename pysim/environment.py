import math
from abc import ABC

import time
import pybullet
import random
import numpy as np

from collections import deque
from pybullet_envs.bullet import bullet_client
from gym import spaces
from pysim.constants import *

from pysim import track
from pysim import agent
from pysim import positions
from pysim.utils import get_reward_function
from sac_torch.utils import get_helper_logger


class CrazyCar(ABC):

    def __init__(self,
                 date=None,
                 track_id=1,
                 renders=False,
                 origin=None,
                 reward_name="theta"):

        if origin is None:
            origin = [0, 0, 0]

        self._timeStep = 0.01
        self._actionRepeat = ACTION_REP
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = DISCRETE_ACTION
        self._origin = origin
        self._collisionCounter = 0
        self._poscar = positions.CarPosition(origin)
        self._reward_function = get_reward_function()[reward_name]
        self._speed = 0
        self._track_id = track_id
        self.logger = None

        if self._renders:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        if OBSERVATION_TYPE == 'sensor+image':
            state, obs = self.reset([2.9 - 0.7/2, 1.1, math.pi/2])
        else:
            obs = self.reset([2.9 - 0.7/2, 1.1, math.pi/2])

        # define observation space
        observationDim = obs.shape
        observation_high = np.full(observationDim, 1)
        observation_low = np.zeros(observationDim)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        # define state space
        if OBSERVATION_TYPE == 'sensor+image':
            stateDim = state.shape
            state_high = np.full(stateDim, 1)
            state_low = np.zeros(stateDim)
            self.state_space = spaces.Box(state_low, state_high, dtype=np.float32)

        # define action space
        if self._isDiscrete:
            self.action_space = spaces.Discrete(9)
        else:
            action_low = np.array([0.5, -1])
            action_high = np.array([1, 1])

            self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        if date is not None:
            self.logger = get_helper_logger(self.__class__.__name__, date)
            self.logger.info(f'{str(self.__class__.__name__)}-Environment has created')
            self.logger.info(f'Reward function: {str(self._reward_function)}')

            if hasattr(self, 'state_space'):
                self.logger.info(f'State shape: {str(self.state_space.shape)}')

            self.logger.info(f'Observation shape: {str(self.observation_space.shape)}')
            self.logger.info(f'Action shape: {str(self.action_space.shape)}')
            self.logger.info(f'Track id: {str(self._track_id)}')

    def _reset(self):
        if self.logger is not None:
            self.logger.info("Environment has reset")
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 60., numSolverIterations=550, numSubSteps=8)

        # time
        self._lastTime = time.time()

        # spawn plane
        self._planeId = self._p.loadURDF("./pysim/data/plane.urdf")

        # spawn race track
        self._direction_field, self.wall_ids = track.createRaceTrack(self._p, self._origin, track_id=self._track_id)

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

        self._racecar = agent.Racecar(self._p, self._origin, carPos, self._planeId,
                                      timeStep=self._timeStep, direction_field=self._direction_field, wall_ids=self.wall_ids)

        # get observation
        if hasattr(self, 'state_space'):
            state, obs = self._racecar.getObservation()
            return state, obs
        obs = self._racecar.getObservation()
        return np.array(obs)

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

        self._envStepCounter += 1

        reward = self._reward()
        done = self._termination()

        if hasattr(self, 'state_space'):
            state, obs = self._racecar.getObservation()
            return np.array(state), np.array(obs), reward, done, {}
        obs = self._racecar.getObservation()
        return np.array(obs), reward, done, {}

    def _termination(self):
        return self._envStepCounter > MAX_STEPS or self._terminate or self._racecar.atGoal

    def _reward(self):

        diffAngle = self._racecar.diffAngle()
        # angleField = self._racecar.getAngleField()

        # cross road
        # isCross = angleField in list(range(45, 360, 90))

        # sensors
        # sensors = self._racecar.getSensor()

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

    def report(self):
        return self._racecar.nCollision


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

        # difference level
        carPos1 = [2.9 - 0.7 / 2, 1.5, math.pi / 2.0]
        carPos2 = [2.9 - 0.7 / 2, 1.2, math.pi / 2.0]

        # same level
        carPos3 = [2.3, 1.2, math.pi/2]
        carPos4 = [2.7, 1.2, math.pi / 2]

        pos = [[carPos1, carPos2], [carPos2, carPos1], [carPos3, carPos4], [carPos4, carPos3]]

        np.random.shuffle(pos)

        # 2 agent
        self._racecars = [
            agent.Racecar(self._p, self._origin, carPos, self._planeId, \
                          timeStep=self._timeStep, direction_field=self._direction_field)
            for carPos in pos[0]
        ]

        for i in range(2):
            print(f"Agent {i} at position {pos[0][i][:-1]}")

        return self.getObservationAll()

    def getObservationAll(self):
        return np.array([racecar.getObservation() for racecar in self._racecars])

    def _termination(self):
        return self._envStepCounter > MAX_STEPS or self._terminate or \
               (self._racecars[0].atGoal or self._racecars[1].atGoal) or \
               self._racecars[0].nCollision == 1 or self._racecars[1].nCollision == 1

    def step(self, action):

        rewards = []
        obses = []

        for car_id, (racecar, realaction) in enumerate(zip(self._racecars, action)):

            racecar.applyAction(realaction)

            obs = racecar.getObservation()
            reward = self._reward(racecar, realaction, obs)

            if racecar.atGoal:
                reward += 1000
                self.atGoal = car_id

            rewards.append(reward)
            obses.append(obs)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()

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
        self.logger = self.env.logger

        if self.state_space is not None:
            self.states = deque([], maxlen=k)

        if self.logger is not None:
            self.logger.info('Use Stack Frame environment')

            if hasattr(self.env, 'state_space'):
                self.logger.info(f'State shape: {str(self.state_space.shape)}')

            self.logger.info(f'New Observation shape: {str(self.observation_space.shape)}')
            self.logger.info(f'New Action shape: {str(self.action_space.shape)}')

    @property
    def state_space(self):
        if hasattr(self.env, 'state_space'):
            shape = self.env.state_space.shape
            shape = (shape[0] * self.k, )
            state_high = np.full(shape, 1)
            state_low = np.zeros(shape)
            return spaces.Box(state_low, state_high, dtype=np.float32)
        return None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        shape = self.env.observation_space.shape
        if len(shape) != 1:  # image
            shape = shape[:-1] + (self.k, )
        else:  # sensor
            shape = (shape[0] * self.k, )
        observation_high = np.full(shape, 1)
        observation_low = np.zeros(shape)
        return spaces.Box(observation_low, observation_high, dtype=np.float32)

    @property
    def _reward_function(self):
        return self.env._reward_function

    def reset(self, *args, **kwargs):
        if self.state_space is not None:
            state, obs = self.env.reset(*args, **kwargs)
        else:
            obs = self.env.reset(*args, **kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
            if self.state_space is not None:
                self.states.append(state)
        if self.state_space is not None:
            return self._get_state(), self._get_obs()
        return self._get_obs()

    def step(self, action):
        state, obs, rew, done, info = self.env.step(action)
        self.frames.append(obs)
        if self.state_space is not None:
            self.states.append(state)
            return self._get_state(), self._get_obs(), rew, done, info
        return self._get_obs(), rew, done, info

    def _get_state(self):
        return np.concatenate(list(self.states), axis=-1)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)

    def report(self):
        return self.env.report()


if __name__ == '__main__':
    env = CrazyCar(renders=True, track_id=2)
    # env = FrameStack(env)
    # print(env.observation_space.shape)
    # env.reset(random_position=False, PosIndex=6)

    # obs = env.reset(random_position=False)
    state, obs = env.reset([2.5, 6, math.pi*2 / 2.0], random_position=False)
    # env.p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[1.5, 3.3, 0])
    print(state.shape, obs.shape)
    # x - [2.1, 2.9]
    while 1:
        state, obs, rew, done, _ = env.step(np.array([0, 0]))
        # print(obs.shape)
        pass
