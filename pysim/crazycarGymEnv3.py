import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
import time
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
# from . import racecar
from . import crazycar
# import crazycar
import random
# from . import bullet_client
from pybullet_envs.bullet import bullet_client
import pybullet_data
from pkg_resources import parse_version

from . import track
from . import positions

# import track as track

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class CrazycarGymEnv3(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=2,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False,
                 calibration=False,
                 actionRandomized=True,
                 origin=[0, 0, 0]):
        print("init")
        self._timeStep = 0.005
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._ballUniqueId = -1
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._prevAction = -1
        self._lastRandomAction = -1
        self._realCar = None
        if self._renders:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()
        self._calibration = calibration
        self._origin = origin
        self._actionRanomized=actionRandomized
        self._collisionCounter = 0
        self._poscar = positions.CarPosition(origin)

        self.seed()
        # self.reset()
        observationDim = 6
        # print("observationDim")
        # print(observationDim)
        # observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        observation_high = np.ones(observationDim) * 1000  # np.inf
        if (isDiscrete):
            self.action_space = spaces.Discrete(6) #91
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(-observation_high, observation_high, dtype=np.float32)
        self.viewer = None

    def reset(self, newCarPos=None):
        self._p.resetSimulation()
        # p.setPhysicsEngineParameter(numSolverIterations=300)
        self._p.setTimeStep(self._timeStep)

        self._planeId = self._p.loadURDF(os.path.join(currentdir, "data/plane.urdf"))
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/restitution.py
        # p.changeDynamics(planeId,-1,lateralFriction=1)
        # TODO
        track.createRaceTrack(self._p, self._origin)

        # dist = 5 +2.*random.random()
        # ang = 2.*3.1415925438*random.random()

        # ballx = dist * math.sin(ang)
        # bally = dist * math.cos(ang)
        # ballx = originPos[0] + 0.2+random.random()*2.0 #2.9/2
        #ballx = self._origin[0] + 2.9 / 2
        #bally = self._origin[1] + 6.5 - 0.715 / 2
        ballx = self._origin[0] + 0.715 / 2
        bally = self._origin[1] + 4.0 - 0.715 / 2
        ballz = 0.0

        if newCarPos is None:
            if self._envStepCounter < 30000:
                carPos = self._poscar.getNewPosition(random.randint(3, 3))  # self._poscar.len()))
            else:
                carPos = self._poscar.getNewPosition(random.randint(3, 3)) #self._poscar.len()))
        else:
            carPos = newCarPos
        if self._calibration:
            carPos = [2.70, 0.5, 0]

        self._ballUniqueId = self._p.loadURDF(os.path.join(self._urdfRoot, "sphere2.urdf"), [ballx, bally, ballz],
                                              globalScaling=0.60)
        self._p.changeDynamics(self._ballUniqueId, -1, mass=0)
        self._p.setGravity(0, 0, -10)
        self._racecar = crazycar.Racecar(self._p, self._origin, carPos, urdfRootPath=self._urdfRoot,
                                         timeStep=self._timeStep, calibration=self._calibration)
        self._envStepCounter = 0
        self._terminate = False
        self._prevAction = -1
        self._lastRandomAction = -1
        self._collisionCounter = 0


        for i in range(100):
            self._p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = []  # self._racecar.getObservation()
        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        ballpos, ballorn = self._p.getBasePositionAndOrientation(self._ballUniqueId)
        invCarPos, invCarOrn = self._p.invertTransform(carpos, carorn)
        ballPosInCar, ballOrnInCar = self._p.multiplyTransforms(invCarPos, invCarOrn, ballpos, ballorn)

        # print(carorn)
#        self._observation.extend([ballPosInCar[0], ballPosInCar[1]])

        posEuler = self._p.getEulerFromQuaternion(carorn)

        yaw = posEuler[2]
        
        x, y  = carpos[0], carpos[1]

        def rayWithRadians(x, y, radians, R=[3, 3]):

            # calculate position to rayTest
            x_new = R[0] * math.cos(radians) + x
            y_new = R[1] * math.sin(radians) + y

            try:
                # position ray hit
                _, _, _, pos, _ = self._p.rayTest([x, y, 0], [x_new, y_new, 0])[0]

                # distance from car
                distance = ((pos[0] - x) ** 2 + (pos[1] - y) ** 2) ** 0.5

                # track.createObj(self._p, self._origin, pos[0], pos[1])
                
                return x_new, y_new, distance
            except:
                print('not found object')

                return 0, 0, -1

        x_new, y_new, distanceFront = rayWithRadians(x, y, yaw)

        x_new, y_new, distanceLeft  = rayWithRadians(x, y, yaw-math.pi/2)

        x_new, y_new, distanceRight = rayWithRadians(x, y, yaw+math.pi/2)

        self._observation.extend([carpos[0], carpos[1], yaw, distanceFront, distanceLeft, distanceRight])
        return self._observation

    def step(self, action):
        if (self._renders):
            basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
            self._p.resetDebugVisualizerCamera(2.5, -90, -40, basePos)

        if (self._isDiscrete):
            #fwd = [-1,-1,-1,0,0,0,1,1,1]
            #steerings = [-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6]
            fwd = [1, 1, 1, 1, 1, 1]
            steerings = [60, 75, 105, 106, 136, 151]
            forward = fwd[action]
            steer = steerings[action]
            #steerings = range(60, 151, 1)
            #forward = 1
            #if (self._prevAction != action):
            #    self._lastRandomAction = min(max(action + random.randint(-2, 2), 0), len(steerings) - 1)
            #    self._prevAction = action
            #    # print("lastAction = {}".format(self._lastRandomAction))
            #if self._actionRanomized is False or self._realCar is not None:
            #    steer = steerings[action]
            #else:
            #    steer = steerings[self._lastRandomAction]
            #if (self._calibration):
            #    forward = 1
            #    steer = 0
            realaction = [forward, steer]
        else:
            realaction = action

        self._racecar.applyAction(realaction)
        if self._realCar is None:
            for i in range(self._actionRepeat):
                self._p.stepSimulation()
                if self._renders:
                    time.sleep(self._timeStep)
                self._observation = self.getExtendedObservation()

                if self._termination():
                    break
                self._envStepCounter += 1
        else:
            self._realCar.setSteering(realaction[1])
            self._p.stepSimulation()
            self._observation = self.getExtendedObservation()
            self._envStepCounter += 1

        reward = self._reward()
        done = self._termination()
        # print("len=%r" % len(self._observation))
        # print("{}".format(self._observation))

        return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def getCarBasePositionAndOrientation(self):
        return self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

    def getTime(self):
        return self._envStepCounter * self._timeStep

    def _termination(self):
        return self._envStepCounter > 1000 or self._terminate

    def _carCollision(self, carLinkIndex):
        aabbmin, aabbmax = self._p.getAABB(self._racecar.racecarUniqueId,
                                           carLinkIndex)  # 5==red block; 1==right wheel; 3==left wheel
        objs = self._p.getOverlappingObjects(aabbmin, aabbmax)
        # print("planeid={}, ballid={}, carid={}".format(self._planeId, self._ballUniqueId, self._racecar.racecarUniqueId))
        # print(objs)
        for x in objs:
            if (x[1] == -1 and not (
                    x[0] == self._racecar.racecarUniqueId or x[0] == self._ballUniqueId or x[0] == self._planeId)):
                # print("collision with: {}".format(x))
                return True
        return False

    def setRealCar(self, car):
        self._realCar = car

    def resetCarPositionAndOrientation(self, posx, posy, angle):
        base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        posObj = [posx, posy, base_pos[2]]
        ornObj = self._p.getQuaternionFromEuler([0, 0, angle])
        self._p.resetBasePositionAndOrientation(self._racecar.racecarUniqueId, posObj, ornObj)

    def _reward2(self):

        closestPoints = self._p.getClosestPoints(self._racecar.racecarUniqueId, self._ballUniqueId, 10000)

        reward = closestPoints[0][8]

        if self._carCollision(5) or self._carCollision(1) or self._carCollision(3) or self._carCollision(0) or self._carCollision(2) or self._carCollision(4):
            reward = -10000
            self._collisionCounter += 1


        

        if self._collisionCounter > 50:
            self._terminate = True

        if -closestPoints[0][8] >= -0.05:
            self._terminate = True
            reward += 1000000

        return reward

    def _reward(self):

        return self._reward2()

        reward = -1

        closestPoints = self._p.getClosestPoints(self._racecar.racecarUniqueId, self._ballUniqueId, 10000)
        numPt = len(closestPoints)
        penalty = 0
        if self._carCollision(5) or self._carCollision(1) or self._carCollision(3):
            penalty = -10000
            self._collisionCounter += 1

        reward = -1 * closestPoints[0][8] - 0 * self._envStepCounter + penalty

        if -closestPoints[0][8] >= -2.5:
            reward = -0.75 * closestPoints[0][8] - 0 * self._envStepCounter + penalty

        if -closestPoints[0][8] >= -1.5:
            reward = -0.5 * closestPoints[0][8] - 0 * self._envStepCounter + penalty

        if -closestPoints[0][8] >= -0.5:
            reward = -0.25 * closestPoints[0][8] - 0 * self._envStepCounter + penalty

        if self._collisionCounter > 50:
            self._terminate = True
        if -closestPoints[0][8] >= -0.05:
            self._terminate = True
            reward += 1000000

        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step

