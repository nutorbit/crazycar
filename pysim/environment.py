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
from pysim.constants import MAX_STEPS, RENDER_HEIGHT, RENDER_WIDTH, MAX_SPEED, MIN_SPEED, RANDOM_POSITION

from pysim import track
from pysim import crazycar
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
        observationDim = 5
        observation_high = np.full(observationDim, 5)
        observation_low = np.zeros(observationDim)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        # define action space
        if (isDiscrete):
            self.action_space = spaces.Discrete(9) #91
        else:
            action_low  = np.array([-1])
            action_high = np.array([1])

            self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def reset(self, newCarPos=None):
        
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)

        # spawn plane
        self._planeId = self._p.loadURDF("./pysim/data/plane.urdf")

        # spawn race track
        track.createRaceTrack(self._p, self._origin)

        # spawn car
        if newCarPos is None:
            if RANDOM_POSITION:
                carPos = self._poscar.getNewPosition(random.randint(1, 11))
            else:
                carPos = self._poscar.getNewPosition(random.randint(1, 1))
        else:
            carPos = newCarPos

        self._racecar = crazycar.Racecar(self._p, self._origin, carPos, urdfRootPath=self._urdfRoot,
                                         timeStep=self._timeStep, calibration=False)

        self.targetX = carPos[0]
        self.targetY = carPos[1]
        self.start = True

        # reset common variables
        for i in range(100):
            self._p.stepSimulation()

        self._p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        self._terminate = False
        self._collisionCounter = 0
        self._observation = self.getExtendedObservation()

        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation = []
        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

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

                return 0, 0, 5

        x_new, y_new, distanceFront = rayWithRadians(x, y, yaw)

        x_new, y_new, distanceRight  = rayWithRadians(x, y, yaw-math.pi/4)

        x_new, y_new, distanceLeft = rayWithRadians(x, y, yaw+math.pi/4)

        x_new, y_new, distanceRightSide  = rayWithRadians(x, y, yaw-math.pi/2)

        x_new, y_new, distanceLeftSide = rayWithRadians(x, y, yaw+math.pi/2)

        self._observation.extend([distanceLeftSide, distanceLeft, distanceFront, distanceRight, distanceRightSide])
        return self._observation

    def step(self, action):
        if (self._renders):
            basePos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
            self._p.resetDebugVisualizerCamera(2.5, -90, -40, basePos)

        if (self._isDiscrete):
            fwd = [1, 1, 1, 1, 1, 1, 1, 1, 1]
            steerings = [-1, -0.75, -0.45, -0.25, 0.00, 0.25, 0.45, 0.75, 1]
            forward = fwd[action]
            steer = steerings[action]
            realaction = [forward, steer]
        else:
            realaction = [1, action[0]]
            # realaction = action

        self._racecar.applyAction(realaction)

        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self.getExtendedObservation()

            if self._termination():
                break
            self._envStepCounter += 1

        reward = self._reward()
        done   = self._termination()
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

