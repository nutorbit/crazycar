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
from pysim.constants import MAX_STEPS, RENDER_HEIGHT, RENDER_WIDTH, MAX_SPEED, MIN_SPEED, RANDOM_POSITION, DISTANCE_SENSORS

from pysim import track
from pysim import agent
from pysim import positions


class CrazyCar(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=2,
                 isDiscrete=False,
                 renders=False,
                 origin=[0, 0, 0]):

        self._timeStep = 0.005
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._origin = origin
        self._collisionCounter = 0
        self._poscar = positions.CarPosition(origin)
        self._speed = 0

        if self._renders:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self.seed()

        # define observation space
        observationDim = len(DISTANCE_SENSORS)
        observation_high = np.full(observationDim, 10)
        observation_low = np.zeros(observationDim)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        # define action space
        if (isDiscrete):
            self.action_space = spaces.Discrete(9) #91
        else:
            action_low  = np.array([-1])
            action_high = np.array([1])

            self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def _reset(self):

        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)

        # time
        self._lastTime = time.time()

        # spawn plane
        self._planeId = self._p.loadURDF("./pysim/data/plane.urdf")

        # spawn race track
        track.createRaceTrack(self._p, self._origin)

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
                carPos = self._poscar.getNewPosition(random.randint(1, 1))
        else:
            carPos = newCarPos

        self._racecar = agent.Racecar(self._p, self._origin, carPos, urdfRootPath=self._urdfRoot,
                                         timeStep=self._timeStep, calibration=False)
        
        # get observation
        self._observation = self._racecar.getSensor()

        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cameraImage(self):
        ls = self._p.getLinkState(self._racecar.racecarUniqueId, 5, computeForwardKinematics=True)
        camPos = ls[0]
        camOrn = ls[1]
        camMat = self._p.getMatrixFromQuaternion(camOrn)
        upVector = [0,0,1]
        forwardVec = [camMat[0],camMat[3],camMat[6]]
        camUpVec =  [camMat[2],camMat[5],camMat[8]]
        camTarget = [camPos[0]+forwardVec[0]*10,camPos[1]+forwardVec[1]*10,camPos[2]+forwardVec[2]*10]
        camUpTarget = [camPos[0]+camUpVec[0],camPos[1]+camUpVec[1],camPos[2]+camUpVec[2]]
        viewMat = self._p.computeViewMatrix(camPos, camTarget, camUpVec)
        projMat = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)
        return self._p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, renderer=self._p.ER_BULLET_HARDWARE_OPENGL, shadow=0)[2]

    def step(self, action):
        if (self._renders):

            now_time = time.time()
            if now_time-self._lastTime>.3:
                _ = self.cameraImage()

        if (self._isDiscrete):
            fwd = [1, 1, 1, 1, 1, 1, 1, 1, 1]
            steerings = [-1, -0.75, -0.45, -0.25, 0.00, 0.25, 0.45, 0.75, 1]
            forward = fwd[action]
            steer = steerings[action]
            realaction = [forward, steer]
        else:
            realaction = [1, action[0]]
            # realaction = action
        if len(action) == 2:
            realaction = action

        self._racecar.applyAction(realaction)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self._racecar.getSensor()

            if self._termination():
                break
            self._envStepCounter += 1

        reward = self._reward()
        done   = self._termination()

        return np.array(self._observation), reward, done, {}

    def _termination(self):
        return self._envStepCounter > MAX_STEPS or self._terminate

    def _carCollision(self, carLinkIndex):
        aabbmin, aabbmax = self._p.getAABB(self._racecar.racecarUniqueId,
                                           carLinkIndex)  # 5==red block; 1==right wheel; 3==left wheel
        objs = self._p.getOverlappingObjects(aabbmin, aabbmax)
        # print(objs)
        for x in objs:
            if (x[1] == -1 and not (x[0] == self._racecar.racecarUniqueId or x[0] == self._planeId)):
                return True
        return False

    def _reward(self):

        reward = 0

        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

        x, y  = carpos[0], carpos[1]

        if self._carCollision(5) or self._carCollision(1) or self._carCollision(3) or self._carCollision(0) or self._carCollision(2) or self._carCollision(4):
            reward = -1
            self._collisionCounter += 1
            
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
            agent.Racecar(self._p, self._origin, carPos, urdfRootPath=self._urdfRoot, timeStep=self._timeStep, calibration=False)
            for carPos in carPoses 
        ]

        return self.getObservation()

    def getObservation(self):
        return [racecar.getSensor() for racecar in self._racecars]

    def step(self, action):
        # if (self._renders):

        #     now_time = time.time()
        #     if now_time-self._lastTime>.3:
        #         _ = self.cameraImage()

        for racecar, realaction in zip(self._racecars, action):

            racecar.applyAction(realaction)

            for i in range(self._actionRepeat):
                self._p.stepSimulation()

                if self._termination():
                    break
                self._envStepCounter += 1


            reward = self._reward()
            done   = self._termination()

        observation = self.getObservation()

        return observation, reward, done, {}

    def _carCollision(self, carLinkIndex):
        res = []
        for racecar in self._racecars:
            isCheck = False
            aabbmin, aabbmax = self._p.getAABB(racecar.racecarUniqueId,
                                            carLinkIndex)  # 5==red block; 1==right wheel; 3==left wheel
            objs = self._p.getOverlappingObjects(aabbmin, aabbmax)
            # print(objs)
            for x in objs:
                if (x[1] == -1 and not (x[0] == racecar.racecarUniqueId or x[0] == self._planeId)):
                    res.append(True)
                    isCheck = True
            if not isCheck:
                res.append(False)
        return res

    def _reward(self):
        pass