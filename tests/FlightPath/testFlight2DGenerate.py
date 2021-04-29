import sys
sys.path.append('./') #TODO: Installation for bezier curve library

import numpy as np
from Lie import SE2
from Flight import FlightOptParams
from Flight2D import Flight2D
from matplotlib import pyplot as plt
from time import perf_counter

#==[1] Specification

vMin =  5
vMax = 10
maxG = 4
startTime = 3.5
endTime   = 4
tspan = [startTime, endTime]
order = 4


#==[2] Terminal poses.

r   = 0.5*(vMin + vMax)
the =  np.pi/3
phi = -np.pi/2


gi = SE2()
gf = SE2(x=r*np.array([[np.cos(the)],[np.sin(the)]]), R=SE2.rotationMatrix(the+phi))

optS = FlightOptParams(init=5, final=6)

#fp1 = Flight2D(gi, gf, bezierOrder = order, tspan= tspan, optParams=optS)

fp1 = Flight2D()
fp1.setDynConstraints(vMin, vMax, 4)
fp1.generate(startTime, gi, endTime, gf)

tic = perf_counter()
fp1.optimizeBezierPath()
toc = perf_counter()

t = np.linspace(startTime, endTime, 100)


print("Path optimization took: ", toc-tic, " seconds")
fp1.plotControlPoints()
fp1.plotCurve()
plt.title("Curve")

print("Time optimization took: ", toc-tic, " seconds")
x = fp1.evalPos(t)
print("Final X: ", x[0,-1])
plt.figure()
plt.title("Speed")
v = fp1.evalVel(t)

plt.plot(t,np.linalg.norm(v, 2, 0), 'r-')
plt.legend(['Optimized', 'Un-optimized'])
plt.show()