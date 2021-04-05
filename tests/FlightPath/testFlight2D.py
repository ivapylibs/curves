import sys
sys.path.append('./') #TODO: Installation for bezier curve library

import numpy as np
from Lie import SE2
from Flight2D import Flight2D, FlightOptParams
from matplotlib import pyplot as plt
from time import perf_counter

#==[1] Specification

vMin =  5
vMax = 10
maxG = 4
tf   = 2
order = 4


#==[2] Terminal poses.

r   = 0.5*(vMin + vMax)
the =  np.pi/3
phi = -np.pi/2


gi = SE2()
gf = SE2(x=r*np.array([[np.cos(the)],[np.sin(the)]]), R=SE2.rotationMatrix(the+phi))

optS = FlightOptParams(init=5, final=3)

fp1 = Flight2D(gi, gf, order, tf, optParams=optS)
fp2 = Flight2D(gi, gf, order, tf, optParams=optS)



tic = perf_counter()
fp1.optimizeBezierPath()
toc = perf_counter()

fp2.optimizeBezierPath()

t = np.linspace(0, tf, 100)

fp1.bezier.plot()
fp1.plotCurve()
plt.title("Curve")

print("Path optimization took: ", toc-tic, " seconds")

fp1.setDynConstraints(vMin, vMax, maxG)
fp2.setDynConstraints(vMin, vMax, maxG)

tic = perf_counter()
fp1.optimizeTimePoly()
toc = perf_counter()
print("Time optimization took: ", toc-tic, " seconds")

plt.figure()
plt.title("Speed")
v = fp1.evalVel(t)
print(np.cumsum(v[0,:])*(tf/100))

v2 = fp2.evalVel(t)
plt.plot(t,np.linalg.norm(v, 2, 0), 'r-')
plt.plot(t,np.linalg.norm(v2, 2, 0), 'b-')
plt.legend(['Optimized', 'Un-optimized'])
plt.show()