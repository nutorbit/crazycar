import copy
import math
import time

import numpy as np

from skimage.color import rgba2rgb, rgb2gray

from pysim.constants import *
from pysim import track

class Racecar:

    def __init__(self, bullet_client, origin, carpos, planeId, direction_field, urdfRootPath='', timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self._p = bullet_client
        self._origin = origin
        self._carpos = carpos
        self._direction_field = direction_field
        self._dist_sensors = DISTANCE_SENSORS
        self.speed = 0
        self.rayHitColor = [1,0,0]
        self.rayMissColor = [0,1,0]
        self._time = time.time()
        self._sensor = []
        self._planeId = planeId
        self.rayFrom = []
        self.rayTo = []
        self.rayRange = math.pi/3
        self.reset()

    def reset(self):
        x = self._origin[0]
        y = self._origin[1]
        z = self._origin[2] + 0.03
        scale = 0.41
        carsize = 0.205
        carx = x + self._carpos[0] 
        cary = y + self._carpos[1] - carsize/2
        #carx = x+2.90-0.35
        #cary = y+0.35-carsize/2
        #cary = y + 5.0-carsize/2
        carStartOrientation = self._p.getQuaternionFromEuler([0, 0, self._carpos[2]])
        carStartOrientation90 = self._p.getQuaternionFromEuler([0,0,math.pi/2])
        carStartOrientation00 = self._p.getQuaternionFromEuler([0,0,0])
#        carId = self._p.loadURDF("data/racecar/racecar.urdf", [carx, cary, z], carStartOrientation90, globalScaling=scale)
        car = self._p.loadURDF("./pysim/data/racecar/racecar_differential1.urdf", [carx, cary, z], carStartOrientation, globalScaling=scale,useFixedBase=False)
#         print(car)
#         print('---')
# #        car = self._p.loadURDF(os.path.join(self.urdfRootPath,"racecar/racecar_differential.urdf"), [0,0,.2],useFixedBase=False)
        self.racecarUniqueId = car
        #for i in range (self._p.getNumJoints(car)):
        #    print (self._p.getJointInfo(car,i))
        for wheel in range(self._p.getNumJoints(car)):
            self._p.setJointMotorControl2(car,wheel,self._p.VELOCITY_CONTROL,targetVelocity=0,force=0)
            self._p.getJointInfo(car,wheel)

        #self._p.setJointMotorControl2(car,10,self._p.VELOCITY_CONTROL,targetVelocity=1,force=10)
        c = self._p.createConstraint(car,9,car,11,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=1, maxForce=10000)

        c = self._p.createConstraint(car,10,car,13,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car,9,car,13,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car,16,car,18,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=1, maxForce=10000)

        c = self._p.createConstraint(car,16,car,19,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car,17,car,19,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car,1,car,18,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
        c = self._p.createConstraint(car,3,car,19,jointType=self._p.JOINT_GEAR,jointAxis =[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
        self._p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15,maxForce=10000)

        self.steeringLinks = [0,2]
        self.maxForce = 1000
        self.nMotors = 2
        self.motorizedwheels=[8,15]
        self.speedMultiplier = 51
        # self.steeringMultiplier = 0.257729373093 # +/- 14.7668 grad
        self.steeringMultiplier = 0.5

        # self.speedParameter = self._p.addUserDebugParameter('Speed', 0, 2, 1)

        for degree in self._dist_sensors:
            self._sensor.append(self._p.addUserDebugLine([0, 0, 0], [self.rayRange, -math.radians(degree), 0], self.rayHitColor, parentObjectUniqueId=car, parentLinkIndex=4))
            self.rayFrom.append([0, 0, 0])
            self.rayTo.append([self.rayRange, -math.radians(degree), 0])

        _ = self.getSensor()

    def getCoordinate(self):
        carpos, carorn = self._p.getBasePositionAndOrientation(self.racecarUniqueId)

        posEuler = self._p.getEulerFromQuaternion(carorn)

        yaw = posEuler[2]

        return carpos[0], carpos[1], yaw

    def getAngleField(self):

        x, y, yaw = self.getCoordinate()

        angles = list(range(0, 360, 45))

        for angle in angles:
            # skip angle
            if angle not in self._direction_field:
                continue

            # in range
            if any([func(x, y) for func in self._direction_field[angle]]):
                return angle

    def diffAngle(self):

        angleField = self.getAngleField()
        _, _, yaw = self.getCoordinate()

        if angleField == 0:
            return abs(yaw-0)
        if angleField == 45:
            return abs(yaw-math.pi/4)
        if angleField == 90:
            return abs(yaw-math.pi/2)
        if angleField == 135:
            return abs(yaw-math.pi/2-math.pi/4)
        if angleField == 180:
            return abs(abs(yaw)-math.pi)
        if angleField == 225:
            return abs(yaw+math.pi/2+math.pi/4)
        if angleField == 270:
            return abs(yaw+math.pi/2)
        if angleField == 315:
            return abs(yaw+math.pi/4)

    def getSensor(self):
        obs = []

        now_time = time.time()

        disp = True
        if (now_time - self._time) > .3:
            disp = True

        results = self._p.rayTestBatch(self.rayFrom, self.rayTo, 0, parentObjectUniqueId=self.racecarUniqueId, parentLinkIndex=4)
        for i, obj in enumerate(results):
            hitObjectUid = obj[0]
            hitFraction = obj[2]
            hitPosition = obj[3]
            # print(hitPosition)
            if (hitFraction == 1.):
                self._p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor, replaceItemUniqueId=self._sensor[i],
                                   parentObjectUniqueId=self.racecarUniqueId, parentLinkIndex=4)

                dist = self.rayRange
            else:
                localHitTo = [self.rayFrom[i][0] + hitFraction * (self.rayTo[i][0] - self.rayFrom[i][0]),
                              self.rayFrom[i][1] + hitFraction * (self.rayTo[i][1] - self.rayFrom[i][1]),
                              self.rayFrom[i][2] + hitFraction * (self.rayTo[i][2] - self.rayFrom[i][2])]
                self._p.addUserDebugLine(self.rayFrom[i], localHitTo, self.rayHitColor, replaceItemUniqueId=self._sensor[i],
                                   parentObjectUniqueId=self.racecarUniqueId, parentLinkIndex=4)
                dist = (localHitTo[0]**2 + localHitTo[1]**2)**0.5

            obs.append(dist)

        return np.array(obs)

    def getObservation(self):

        observation = None

        if OBSERVATION_TYPE == 'image':
            observation = self.getCameraImage() # to gray scale
        if OBSERVATION_TYPE == 'sensor':
            # observation = np.concatenate([self.getSensor(), np.array([self.speed/self.speedMultiplier])]) # norm
            # observation = self.getSensor()/7
            observation = np.concatenate([self.getSensor(), [self.diffAngle()], np.array([self.speed])]) # norm
        if OBSERVATION_TYPE == 'sensor+image':
            observation = np.concatenate([self.getSensor()/7, self.getCameraImage().flatten()/255])

        return observation

    def getCameraImage(self):

        ls = self._p.getLinkState(self.racecarUniqueId, 5, computeForwardKinematics=True)
        camPos = ls[0]
        camOrn = ls[1]
        camMat = self._p.getMatrixFromQuaternion(camOrn)
        upVector = [0, 0, 1]
        forwardVec = [camMat[0], camMat[3], camMat[6]]
        camUpVec =  [camMat[2], camMat[5], camMat[8]]
        camTarget = [camPos[0]+forwardVec[0]*10, camPos[1]+forwardVec[1]*10, camPos[2]+forwardVec[2]*10]
        camUpTarget = [camPos[0]+camUpVec[0], camPos[1]+camUpVec[1], camPos[2]+camUpVec[2]]
        viewMat = self._p.computeViewMatrix(camPos, camTarget, camUpVec)
        projMat = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0)

        # rgba to gray
        raw = self._p.getCameraImage(CAMERA_WIDTH, CAMERA_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMat, renderer=self._p.ER_TINY_RENDERER, lightColor=[0, 0, 0], shadow=0)[4]
        # print(len(raw))
        # assert 1 != 1
        raw = np.array(raw).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 1))
        raw = np.where(raw > 0, 1, raw)  # segment wall
        # img_rgb = rgba2rgb(raw)
        # img_gray = np.expand_dims(rgb2gray(img_rgb), -1)

        return raw

    def _isCollision(self, part_id):

        aabbmin, aabbmax = self._p.getAABB(self.racecarUniqueId,
                                           part_id)  # 5==red block; 1==right wheel; 3==left wheel
        objs = self._p.getOverlappingObjects(aabbmin, aabbmax)
        # print(objs)

        for x in objs:
            if (x[1] == -1 and not (x[0] == self.racecarUniqueId or x[0] == self._planeId)):
                return True
        return False

    def isCollision(self):

        return any([self._isCollision(i) for i in range(1, 10)])

    def applyAction(self, motorCommands):

        # sp = self._p.readUserDebugParameter(self.speedParameter)
        targetVelocity = motorCommands[0]*self.speedMultiplier
        # targetVelocity = sp*self.speedMultiplier
        self.speed = targetVelocity
        
        steeringAngle = motorCommands[1]*self.steeringMultiplier

        for motor in self.motorizedwheels:
            self._p.setJointMotorControl2(self.racecarUniqueId,motor,self._p.VELOCITY_CONTROL,targetVelocity=targetVelocity,force=self.maxForce)
        for steer in self.steeringLinks:
            self._p.setJointMotorControl2(self.racecarUniqueId,steer,self._p.POSITION_CONTROL,targetPosition=steeringAngle)
