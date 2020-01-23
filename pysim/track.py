import pybullet as p
import time
import math
import pybullet_data
import pygame


def createObj(bullet_client, origin, x, y, z=0):
     p = bullet_client
     startOrientation000 = p.getQuaternionFromEuler([0,0,0])
     elem2900p1 = p.loadURDF("./pysim/data/test.urdf", [x, y, z], startOrientation000)

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


     elem2900p1 = p.loadURDF("./pysim/data/elem2900.urdf", [x, y, z], startOrientation000)
     elem6500p1 = p.loadURDF("./pysim/data/elem6500.urdf", [x + 2.9/2 - width/2, y+6.5/2 + width/2, z], startOrientation090)
     elem2900p2 = p.loadURDF("./pysim/data/elem2900.urdf", [x - width, y+6.5, z], startOrientation000)
     elem6500p2 = p.loadURDF("./pysim/data/elem6500.urdf", [x - 2.9/2 - width/2, y+6.5/2 - width/2, z], startOrientation090)
     # inside1
     elem0200p1 = p.loadURDF("./pysim/data/elem0200.urdf", [x - 1.4/2 - width, y+0.7+0.8 + width/2, z], startOrientation000)
     elem0800p1 = p.loadURDF("./pysim/data/elem0800.urdf", [x - 1.4/2 + width/2, y+0.7+0.8/2, z], startOrientation090)
     elem1400p1 = p.loadURDF("./pysim/data/elem1400.urdf", [x + width, y+0.7+width/2, z], startOrientation000)
     elem5000p1 = p.loadURDF("./pysim/data/elem5000.urdf", [x + width - width/2 + 1.4/2, y+0.7+width +5.0/2, z], startOrientation090)
     elem1400p2 = p.loadURDF("./pysim/data/elem1400.urdf", [x + width - width, y+0.7+width/2+5.0, z], startOrientation000)
     elem2000p1 = p.loadURDF("./pysim/data/elem2000.urdf", [x + width - width/2 - 0.7, y+0.7+5.0-2/2, z], startOrientation090)
     # inside2
     elem2600p1 = p.loadURDF("./pysim/data/elem2600.urdf", [x - 2.9/2 + 1.4 + width/2, y+0.7+width+5.0-width-0.78-2.6/2, z], startOrientation090)
     elem1400p3 = p.loadURDF("./pysim/data/elem1400.urdf", [x - 2.9/2 + 1.4/2, y+0.7+width/2+5.0-2.6/2-0.78-2.6/2, z], startOrientation000)

     # #corners
     elem0720p1 = p.loadURDF("./pysim/data/elem0720.urdf", [x + width + 1.4/2 - d - width/2, y+0.7+width/2+d, z], startOrientation045)
     elem0720p2 = p.loadURDF("./pysim/data/elem0720.urdf", [x + width - 1.4/2 + d - width/2, y+0.7+width/2+d, z], startOrientation135)
     
     x1 = 2.9/2 - 1.4/2 + 0.05/2
     x2 = 2.9/2 - 2.9/2 + 1.4 + 0.05/2
     x3 = 2.9/2 + 0.05 - 0.05/2 + 1.4/2
     x4 = 2.9 - 0.05/2

     y1 = -0.05/2 + 0.7 + 0.05/2
     y2 = -0.05/2 + 0.7 + 0.05/2 + 5.0 - 2.6/2 - 0.78 - 2.6/2
     y3 = -0.05/2 + 0.7 + 5.0 - 2 + 0.05/2
     y4 = -0.05/2 + 0.7 + 0.05 + 5.0 - 0.78 - 0.05
     y5 = -0.05/2 + 0.7 + 0.05/2 + 5
     y6 = -0.05/2 + 6.5
     
     # X-axis
     # createObj(p, origin, x1, 0, 0) # 1
     # createObj(p, origin, x2, 0, 0) # 2
     # createObj(p, origin, x3, 0, 0) # 3
     # createObj(p, origin, x4, 0, 0) # 4

     # # Y-axis
     # createObj(p, origin, 0, y1, 0) # 1
     # createObj(p, origin, 0, y2, 0) # 2
     # createObj(p, origin, 0, y3, 0) # 3
     # createObj(p, origin, 0, y4, 0) # 4
     # createObj(p, origin, 0, y5, 0) # 5
     # createObj(p, origin, 0, y6, 0) # 6


     direction_field = { # degree
          0: [
               lambda i, j: (0 < i < x3) and (0 < j < y1),
               lambda i, j: (0 < i < x1) and (y2 < j < y3),
               lambda i, j: (x1 < i < x2) and (y4 < j < y5)
          ],
          90: [
               lambda i, j: (x3 < i < x4) and (0 < j < y5),
               lambda i, j: (x1 < i < x2) and (y2 < j < y4)
          ],
          -90: [
               lambda i, j: (0 < i < x1) and (y1 < j < y2),
               lambda i, j: (0 < i < x1) and (y3 < j < y6),
               lambda i, j: (x2 < i < x3) and (y2 < j < y5)
          ],
          180: [
               lambda i, j: (x1 < i < x3) and (y1 < j < y2),
               lambda i, j: (x1 < i < x4) and (y5 < j < y6)
          ]
     }

     return direction_field

