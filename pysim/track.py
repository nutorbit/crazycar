import pybullet as p
import time
import math
import pybullet_data
import pygame
from pysim.maps import Map


def createObj(bullet_client, origin, x, y, z=0):
    p = bullet_client
    startOrientation000 = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("./pysim/data/test.urdf", [x, y, z], startOrientation000)


def createRaceTrack(bullet_client, origin, track_id=1):

    m = Map(bullet_client, origin)

    if track_id == 1:
        direction_field = m.map1()

    if track_id == 2:
        direction_field = m.map2()

    return direction_field
