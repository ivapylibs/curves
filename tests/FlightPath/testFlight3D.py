import sys
sys.path.append('./') #TODO: Installation for bezier curve library

import numpy as np
from Lie import SE3
from Flight import Flight, FlightOptParams
from matplotlib import pyplot as plt
from time import perf_counter

#==[1] Specification

vMin =  5
vMax = 10
maxG = 4
tf   = 3
order = 4


#==[2] Terminal poses.

r   = 0.5*(vMin + vMax)
the =  np.pi/3
phi = -np.pi/2


gi = SE3()
#gf = SE2(x=r*np.array([[np.cos(the)],[np.sin(the)]]), R=SE2.rotationMatrix(the+phi))


roll = np.pi/3
pitch = np.pi/4
yaw = np.pi/6
R = np.matmul(np.matmul(SE3.RotZ(yaw), SE3.RotY(pitch)), SE3.RotX(roll))

x = np.array([[3], [6], [2]])
gf = SE3(R=R, x=x)

optS = FlightOptParams(init=5, final=3)

fp1 = Flight(gi, gf, order, tf, optParams=optS)
fp2 = Flight(gi, gf, order, tf, optParams=optS)



tic = perf_counter()
fp1.optimizeBezierPath()
toc = perf_counter()

fp2.optimizeBezierPath()

t = np.linspace(0, tf, 100)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#fp1.bezier.plot(ax)
fp1.plotControlPoints(ax)
fp1.plotCurve(ax)
plt.title("Curve")

print("Path optimization took: ", toc-tic, " seconds")

fp1.setDynConstraints(vMin, vMax, maxG)
fp2.setDynConstraints(vMin, vMax, maxG)

tic = perf_counter()
fp1.optimizeTimePoly()
toc = perf_counter()
print("Time optimization took: ", toc-tic, " seconds")
x = fp1.evalPos(t)
print("Final X: ", x[0,-1])
plt.figure()
plt.title("Speed")
v = fp1.evalVel(t)

v2 = fp2.evalVel(t)
plt.plot(t,np.linalg.norm(v, 2, 0), 'r-')
plt.plot(t,np.linalg.norm(v2, 2, 0), 'b-')
plt.legend(['Optimized', 'Un-optimized'])
plt.show()