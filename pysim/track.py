import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import pybullet as p
import time
import math
import pybullet_data
import pygame

def createRaceCar(bullet_client, origin):
     p = bullet_client
     x = origin[0]
     y = origin[1]
     z = origin[2] +0.03
     scale = 0.41
     carsize = 0.205
     carx = x+2.90-0.35
     cary = y+0.35-carsize/2
     carStartOrientation90 = p.getQuaternionFromEuler([0,0,math.pi/2])
     carStartOrientation00 = p.getQuaternionFromEuler([0,0,0])
#     carId = p.loadURDF("data/racecar/racecar.urdf", [carx, cary, z], carStartOrientation90, globalScaling=scale)
     carId = p.loadURDF(os.path.join(currentdir, "data/racecar/racecar_differential.urdf"), [carx, cary, z], carStartOrientation90, globalScaling=scale)
## Test car size!
#     carId = p.loadURDF("data/racecar/racecar.urdf", [carx, cary, z], carStartOrientation90, globalScaling=scale)
##     p.resetDebugVisualizerCamera(1,-90,-0, [carx, cary, z])
#     p.resetDebugVisualizerCamera(0.3,0,-89, [carx, cary, z])
#     p.loadURDF("data/test0140.urdf", [carx-0.08, cary+0.07, z] , carStartOrientation90)  # axes distance
#     p.loadURDF("data/test0107.urdf", [carx, cary-0.04, z] , carStartOrientation00)       # width
#     p.loadURDF("data/test0205.urdf", [carx+0.08, cary+0.07, z] , carStartOrientation90)   # length
     #color = [1, 0, 0]
     #width = 1
     #p.addUserDebugLine([carx, cary, z], [0,0,0], color, width, 0)
     return carId

def createObj(bullet_client, origin, x, y, z=0):
     p = bullet_client
     startOrientation000 = p.getQuaternionFromEuler([0,0,0])
     elem2900p1 = p.loadURDF(os.path.join(currentdir, "data/test.urdf"), [x, y, z], startOrientation000)

def createRaceTrack(bullet_client, origin):
     p = bullet_client
     width = 0.05
     #x = -2.9/2 + 2.9/2
     #y = -width/2 - 3.5
     #z = 0.1
     x = origin[0] + 2.9/2
     y = origin[1] - width/2.0
     z = origin[2] + 0.1
     d = 0.2545
     width=0.05
     startOrientation000 = p.getQuaternionFromEuler([0,0,0])
     startOrientation045 = p.getQuaternionFromEuler([0,0,math.pi/4])
     startOrientation090 = p.getQuaternionFromEuler([0,0,math.pi/2])
     startOrientation135 = p.getQuaternionFromEuler([0,0,-math.pi/4])
     startPos = [x, y, z]

     #test
     # sensor     = p.loadURDF(os.path.join(currentdir, "data/sensor.urdf"), [x, y, z], startOrientation135)
     # sensor     = p.loadSDF(os.path.join(currentdir, "data/sensor.sdf"))


     elem2900p1 = p.loadURDF(os.path.join(currentdir, "data/elem2900.urdf"), [x, y, z], startOrientation000)
     elem6500p1 = p.loadURDF(os.path.join(currentdir, "data/elem6500.urdf"), [x + 2.9/2 - width/2, y+6.5/2 + width/2, z], startOrientation090)
     elem2900p2 = p.loadURDF(os.path.join(currentdir, "data/elem2900.urdf"), [x - width, y+6.5, z], startOrientation000)
     elem6500p2 = p.loadURDF(os.path.join(currentdir, "data/elem6500.urdf"), [x - 2.9/2 - width/2, y+6.5/2 - width/2, z], startOrientation090)
     # inside1
     elem0200p1 = p.loadURDF(os.path.join(currentdir, "data/elem0200.urdf"), [x - 1.4/2 - width, y+0.7+0.8 + width/2, z], startOrientation000)
     elem0800p1 = p.loadURDF(os.path.join(currentdir, "data/elem0800.urdf"), [x - 1.4/2 + width/2, y+0.7+0.8/2, z], startOrientation090)
     elem1400p1 = p.loadURDF(os.path.join(currentdir, "data/elem1400.urdf"), [x + width, y+0.7+width/2, z], startOrientation000)
     elem5000p1 = p.loadURDF(os.path.join(currentdir, "data/elem5000.urdf"), [x + width - width/2 + 1.4/2, y+0.7+width +5.0/2, z], startOrientation090)
     elem1400p2 = p.loadURDF(os.path.join(currentdir, "data/elem1400.urdf"), [x + width - width, y+0.7+width/2+5.0, z], startOrientation000)
     elem2000p1 = p.loadURDF(os.path.join(currentdir, "data/elem2000.urdf"), [x + width - width/2 - 0.7, y+0.7+5.0-2/2, z], startOrientation090)
     #inside2
     elem2600p1 = p.loadURDF(os.path.join(currentdir, "data/elem2600.urdf"), [x - 2.9/2 + 1.4 + width/2, y+0.7+width+5.0-width-0.78-2.6/2, z], startOrientation090)
     elem1400p3 = p.loadURDF(os.path.join(currentdir, "data/elem1400.urdf"), [x - 2.9/2 + 1.4/2, y+0.7+width/2+5.0-2.6/2-0.78-2.6/2, z], startOrientation000)

     #corners
     elem0720p1 = p.loadURDF(os.path.join(currentdir, "data/elem0720.urdf"), [x + width + 1.4/2 - d - width/2, y+0.7+width/2+d, z], startOrientation045)
     elem0720p2 = p.loadURDF(os.path.join(currentdir, "data/elem0720.urdf"), [x + width - 1.4/2 + d - width/2, y+0.7+width/2+d, z], startOrientation135)
     
     return [x, y, z]

