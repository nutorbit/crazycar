import math

import numpy as np

from crazycar.agents.constants import DISTANCE_SENSORS, CAMERA_WIDTH, CAMERA_HEIGHT
from crazycar.utils import rgba2rgb, rgb2gray, timing


class BaseAgent:
    """
    The Racecar

    Args:
        bullet_client: bullet client
        origin: origin of car
        carpos: position of the car [x, y, angle (radians)]
        plane_id: plane for put the car to
        direction_field: define the direction
        wall_ids: which id is wall
    """

    def __init__(self, bullet_client, origin, carpos, plane_id, direction_field, wall_ids):
        self._p = bullet_client
        self._origin = origin
        self._carpos = carpos
        self._direction_field = direction_field
        self._dist_sensors = DISTANCE_SENSORS
        self.speed = 0
        self.rayHitColor = [1, 0, 0]
        self.rayMissColor = [0, 1, 0]
        self._sensor = []
        self.plane_id = plane_id
        self.rayFrom = []
        self.rayTo = []
        self.rayRange = math.pi / 3
        self.steeringLinks = [0, 2]
        self.maxForce = 1000
        self.motorizedwheels = [8, 15]
        self.speedMultiplier = 100
        self.steeringMultiplier = 0.5
        self.atGoal = False
        self.nCollision = 0
        self.wall_ids = wall_ids
        self.initial()

    @timing('initial_car', debug=False)
    def initial(self):
        """
        Initial the car
        """

        x = self._origin[0]
        y = self._origin[1]
        z = self._origin[2] + 0.03

        scale = 0.41
        carsize = 0.205
        carx = x + self._carpos[0]
        cary = y + self._carpos[1] - carsize / 2
        carStartOrientation = self._p.getQuaternionFromEuler([0, 0, self._carpos[2]])
        car = self._p.loadURDF("./crazycar/data/racecar/racecar_differential1.urdf", [carx, cary, z], carStartOrientation,
                               globalScaling=scale, useFixedBase=False)
        self.racecarUniqueId = car

        # setup wheels
        for wheel in range(self._p.getNumJoints(car)):
            self._p.setJointMotorControl2(car, wheel, self._p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self._p.getJointInfo(car, wheel)

        # setup constraint
        c = self._p.createConstraint(car, 9, car, 11, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._p.createConstraint(car, 10, car, 13, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car, 9, car, 13, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car, 16, car, 18, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=1, maxForce=10000)

        c = self._p.createConstraint(car, 16, car, 19, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car, 17, car, 19, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, maxForce=10000)

        c = self._p.createConstraint(car, 1, car, 18, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
        c = self._p.createConstraint(car, 3, car, 19, jointType=self._p.JOINT_GEAR, jointAxis=[0, 1, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

        # setup sensor
        for degree in self._dist_sensors:
            from_ = [0, 0, 0]
            to_   = [np.cos(math.radians(degree)) * self.rayRange, np.sin(math.radians(degree)) * self.rayRange, 0]
            self._sensor.append(
                self._p.addUserDebugLine(from_, to_, self.rayHitColor,
                                         parentObjectUniqueId=car, parentLinkIndex=4))
            self.rayFrom.append(from_)
            self.rayTo.append(to_)
        # print(self.rayTo)
        # _ = self.getSensor()

    def remove_sensor(self):
        """
        Remove car sensor
        """

        for uid in self._sensor:
            # self._p.removeBody(uid)
            self._p.removeUserDebugItem(uid)

    def get_coordinate(self):
        """
        Get coordinate of the car

        Returns:
            x, y, angle (Radians)
        """

        carpos, carorn = self._p.getBasePositionAndOrientation(self.racecarUniqueId)

        posEuler = self._p.getEulerFromQuaternion(carorn)

        yaw = posEuler[2]

        return carpos[0], carpos[1], yaw

    def get_angle_field(self):
        """
        Get the angle of direction to go

        Returns:
            angle in Degrees
        """

        x, y, yaw = self.get_coordinate()

        angles = list(range(-180, 180+1, 45))

        for angle in angles:
            # skip angle
            if angle not in self._direction_field:
                continue

            # in range
            if any([func(x, y) for func in self._direction_field[angle]]):
                return angle

        # return res[0]

    def get_diff_angle(self):
        """
        Get the difference angle between car and direction to go

        Returns:
             absolute angle in Radians
        """

        angleField = self.get_angle_field()
        _, _, yaw = self.get_coordinate()
        # diff = abs((-np.radians(angleField) - yaw)) if yaw <= 0 else abs((np.radians(angleField) - yaw))
        diff = np.radians(angleField) - yaw
        # TODO: fix bug angle
        # print(angleField, np.degrees(yaw), np.degrees(diff), )
        return diff / np.pi

    def get_sensor(self):
        """
        Get sensor around the car

        Returns:
            list of each sensors
        """

        obs = []

        results = self._p.rayTestBatch(self.rayFrom, self.rayTo, 0, parentObjectUniqueId=self.racecarUniqueId,
                                       parentLinkIndex=4)
        for i, obj in enumerate(results):
            hitObjectUid = obj[0]
            hitFraction = obj[2]
            hitPosition = obj[3]
            # print(hitPosition)
            if (hitFraction == 1.):
                self._p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor,
                                         replaceItemUniqueId=self._sensor[i],
                                         parentObjectUniqueId=self.racecarUniqueId, parentLinkIndex=4)

                dist = self.rayRange
            else:
                localHitTo = [self.rayFrom[i][0] + hitFraction * (self.rayTo[i][0] - self.rayFrom[i][0]),
                              self.rayFrom[i][1] + hitFraction * (self.rayTo[i][1] - self.rayFrom[i][1]),
                              self.rayFrom[i][2] + hitFraction * (self.rayTo[i][2] - self.rayFrom[i][2])]
                self._p.addUserDebugLine(self.rayFrom[i], localHitTo, self.rayHitColor,
                                         replaceItemUniqueId=self._sensor[i],
                                         parentObjectUniqueId=self.racecarUniqueId, parentLinkIndex=4)
                dist = (localHitTo[0] ** 2 + localHitTo[1] ** 2) ** 0.5

            obs.append(dist)

        return np.array(obs) / self.rayRange

    @timing('process_camera', debug=False)
    def get_camera(self):
        """
        Get camera image

        Returns:
            image from front camera
        """

        ls = self._p.getLinkState(self.racecarUniqueId, 5, computeForwardKinematics=True)
        camPos = ls[0]
        camOrn = ls[1]
        camOrn = list(camOrn)
        # TODO: add back camera
        camOrn = tuple(camOrn)
        camMat = self._p.getMatrixFromQuaternion(camOrn)
        forwardVec = [camMat[0], camMat[3], camMat[6]]
        camUpVec = [camMat[2], camMat[5], camMat[8]]
        camTarget = [camPos[0] + forwardVec[0] * 10, camPos[1] + forwardVec[1] * 10, camPos[2] + forwardVec[2] * 10]
        viewMat = self._p.computeViewMatrix(camPos, camTarget, camUpVec)
        projMat = (
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128,
        0.0)

        # segment
        # raw = self._p.getCameraImage(CAMERA_WIDTH, CAMERA_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMat,
        #                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL, lightColor=[0, 0, 0], shadow=0)[4]
        # raw = np.array(raw).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 1))
        # raw = np.where(np.isin(raw, self.wall_ids), 1, np.where(raw > max(self.wall_ids), 4, raw))  # segment wall and another car

        # gray image from rgba image
        raw = self._p.getCameraImage(CAMERA_WIDTH, CAMERA_HEIGHT, viewMatrix=viewMat, projectionMatrix=projMat,
                                     renderer=self._p.ER_BULLET_HARDWARE_OPENGL)[2]
        raw = np.array(raw).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))

        raw = rgba2rgb(raw)
        # np.save('./test1.npy', raw)
        raw = np.expand_dims(rgb2gray(raw), -1)
        # np.save('./test2.npy', raw)

        return np.array([raw])

    def _is_collision(self, part_id):
        """
        Helper function for checking the collision

        Args:
            part_id: each part in the car

        Returns:
            True or False
        """

        aabbmin, aabbmax = self._p.getAABB(self.racecarUniqueId,
                                           part_id)  # 5==red block; 1==right wheel; 3==left wheel
        objs = self._p.getOverlappingObjects(aabbmin, aabbmax)

        for x in objs:
            if x[1] == -1 and not (x[0] == self.racecarUniqueId or x[0] == self.plane_id):
                return True
        return False

    def is_collision(self):
        """
        Check this car is collision
        """

        return any([self._is_collision(i) for i in range(1, 10)])

    def apply_action(self, motorCommands):
        """
        Apply action to car

        Args:
            motorCommands: [speed, angle]
        """

        targetVelocity = motorCommands[0] * self.speedMultiplier
        self.speed = motorCommands[0]

        steeringAngle = motorCommands[1] * self.steeringMultiplier

        for motor in self.motorizedwheels:
            self._p.setJointMotorControl2(self.racecarUniqueId, motor, self._p.VELOCITY_CONTROL,
                                          targetVelocity=targetVelocity, force=self.maxForce)
        for steer in self.steeringLinks:
            self._p.setJointMotorControl2(self.racecarUniqueId, steer, self._p.POSITION_CONTROL,
                                          targetPosition=steeringAngle)

        # update
        x, y, _ = self.get_coordinate()
        self.atGoal |= 2.1 <= x <= 2.9 and 0.9 <= y <= 1

    def get_reward(self):
        """
        Define how the reward function calculate

        Returns:
            reward
        """

        raise NotImplementedError

    def get_observation(self):
        """
        Get observation of car

        Returns:
            dictionary
        """

        return NotImplementedError
