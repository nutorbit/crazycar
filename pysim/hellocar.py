import pybullet as p
import time
import math
import pybullet_data
import pygame

def initJoystick():
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joystick controller found")
        exit(-1)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick

def readJoystick(joystick):
    pygame.event.get()
    axes = joystick.get_numaxes()
    targetVelocity = 0
    steeringAngle = 0
    for i in range(axes):
        axis = joystick.get_axis(i)
        if(i==1):
            targetVelocity = axis/-1.0 * 100
        if(i==0):
            steeringAngle =  axis*-0.2577; # == +/-14.76grad ==> steering 60! #axis/(-math.pi/4)            
    return [targetVelocity, steeringAngle]
    
    
def releaseJoystick(joystick):
    joystick.quit()
    pygame.quit()
    
def createRaceCar(origin):
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
     carId = p.loadURDF("data/racecar/racecar_differential.urdf", [carx, cary, z], carStartOrientation90, globalScaling=scale)
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



def createRaceTrack(origin):
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
     elem2900p1 = p.loadURDF("data/elem2900.urdf", [x, y, z], startOrientation000)
     elem6500p1 = p.loadURDF("data/elem6500.urdf", [x + 2.9/2 - width/2, y+6.5/2 + width/2, z], startOrientation090)
     elem2900p2 = p.loadURDF("data/elem2900.urdf", [x - width, y+6.5, z], startOrientation000)
     elem6500p2 = p.loadURDF("data/elem6500.urdf", [x - 2.9/2 - width/2, y+6.5/2 - width/2, z], startOrientation090)
     # inside1
     elem0200p1 = p.loadURDF("data/elem0200.urdf", [x - 1.4/2 - width, y+0.7+0.8 + width/2, z], startOrientation000)
     elem0800p1 = p.loadURDF("data/elem0800.urdf", [x - 1.4/2 + width/2, y+0.7+0.8/2, z], startOrientation090)
     elem1400p1 = p.loadURDF("data/elem1400.urdf", [x + width, y+0.7+width/2, z], startOrientation000)
     elem5000p1 = p.loadURDF("data/elem5000.urdf", [x + width - width/2 + 1.4/2, y+0.7+width +5.0/2, z], startOrientation090)
     elem1400p2 = p.loadURDF("data/elem1400.urdf", [x + width - width, y+0.7+width/2+5.0, z], startOrientation000)
     elem2000p1 = p.loadURDF("data/elem2000.urdf", [x + width - width/2 - 0.7, y+0.7+5.0-2/2, z], startOrientation090)
     #inside2
     elem2600p1 = p.loadURDF("data/elem2600.urdf", [x - 2.9/2 + 1.4 + width/2, y+0.7+width+5.0-width-0.78-2.6/2, z], startOrientation090)
     elem1400p3 = p.loadURDF("data/elem1400.urdf", [x - 2.9/2 + 1.4/2, y+0.7+width/2+5.0-2.6/2-0.78-2.6/2, z], startOrientation000)

     #corners
     elem0720p1 = p.loadURDF("data/elem0720.urdf", [x + width + 1.4/2 - d - width/2, y+0.7+width/2+d, z], startOrientation045)
     elem0720p2 = p.loadURDF("data/elem0720.urdf", [x + width - 1.4/2 + d - width/2, y+0.7+width/2+d, z], startOrientation135)
     
     return [x, y, z]

joystick = initJoystick()
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
wheelVelocityId = p.addUserDebugParameter("wheelVelocity",-100,100,0)
steeringAngleId = p.addUserDebugParameter("steeringAngle",-math.pi/4,math.pi/4,0)


p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.81)
planeId = p.loadURDF("data/plane.urdf")

#https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/restitution.py
p.changeDynamics(planeId,-1,lateralFriction=1)

#originPos = [ 0, 0, 0 ]
originPos = [ -2.9/2, -3.5, 0 ]
createRaceTrack(originPos)
carId = createRaceCar(originPos)

cubePos, cubeOrn = p.getBasePositionAndOrientation(planeId)
#p.resetDebugVisualizerCamera(15,-90,-40,cubePos)

# p.getJointInfo(carId, 0..getNumJoints(carId)) # get Info
#inactiveWheels = [2, 3, 5, 7]
#for wheel in inactiveWheels:
#    p.setJointMotorControl2(carId, wheel, controlMode=p.VELOCITY_CONTROL, targetVelocity=0, force=0)
#steeringLinks = [4 , 6]
#motorizedwheels = [2]
#FROM https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/racecar_differential.py
for i in range (p.getNumJoints(carId)):
    print (p.getJointInfo(carId, i))
for wheel in range(p.getNumJoints(carId)):
    p.setJointMotorControl2(carId, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    p.getJointInfo(carId, wheel)    
    
c = p.createConstraint(carId,9,carId,11,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=1, maxForce=10000)

c = p.createConstraint(carId,10,carId,13,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(carId,9,carId,13,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(carId,16,carId,18,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=1, maxForce=10000)


c = p.createConstraint(carId,16,carId,19,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(carId,17,carId,19,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, maxForce=10000)

c = p.createConstraint(carId,1,carId,18,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15, maxForce=10000)
c = p.createConstraint(carId,3,carId,19,jointType=p.JOINT_GEAR,jointAxis=[0,1,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
p.changeConstraint(c,gearRatio=-1, gearAuxLink = 15,maxForce=10000)

    
steeringLinks   = [0,  2]
motorizedwheels = [8, 15]

carPos, carOrn = p.getBasePositionAndOrientation(carId)
distance=5
yaw = 0
#p.resetDebugVisualizerCamera(distance,yaw,-20,carPos)


steeringAngle = 0.35 #0.78
maxForce = 1000
targetVelocity= 20
mode = p.VELOCITY_CONTROL
#p.setJointMotorControl2(carId, 1, controlMode=mode, force=maxForce)
#p.setJointMotorControl2(bodyUniqueId=carId, jointIndex=00, 
#    controlMode=p.VELOCITY_CONTROL, targetVelocity = targetVel, force = maxForce)
for i in range (10000):
    p.stepSimulation()
    for motor in motorizedwheels:
        p.setJointMotorControl2(carId, motor , controlMode=p.VELOCITY_CONTROL, targetVelocity = targetVelocity, force = maxForce)
#    p.setJointMotorControl2(carId, motor , controlMode=p.VELOCITY_CONTROL, targetVelocity = targetVelocity, force = maxForce)
    for steer in steeringLinks:
        p.setJointMotorControl2(carId, steer, controlMode=p.POSITION_CONTROL, targetPosition = steeringAngle)
        
    [targetVelocity, steeringAngle] = readJoystick(joystick)
    #targetVelocity = p.readUserDebugParameter(wheelVelocityId)
    #steeringAngle = p.readUserDebugParameter(steeringAngleId)
    
    
    carPos, carOrn = p.getBasePositionAndOrientation(carId) 
    
    distance=0.5
    yaw = 0    

    #matrix = p.getMatrixFromQuaternion(carOrn)
    #print(matrix)
    yaw = posEuler[2]*180/math.pi - 90;
    p.resetDebugVisualizerCamera(distance, yaw, -40,carPos)
    time.sleep(1./240.)
    pos, orn = p.getBasePositionAndOrientation(carId)
    #print(pos,orn)
p.disconnect()
releaseJoystick(joystick)
