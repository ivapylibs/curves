import numpy as np
from Lie.group.SE2 import Homog as SE2
from Curves.Flight2D import Flight2D
from Curves.Flight import FlightOptParams
from matplotlib import pyplot as plt
from time import perf_counter

#==[1] Specification

vMin =  5
vMax = 10
maxG = 4
startTime = 0
endTime   = 2
tspan = [startTime, endTime]
order = 4


#==[2] Terminal poses.

r   = 0.5*(vMin + vMax)
the =  np.pi/5
phi = -np.pi/4


gi = SE2()
gf = SE2(x=r*np.array([[np.cos(the)],[np.sin(the)]]), R=SE2.rotationMatrix(the+phi))

optS = FlightOptParams(init=5, final=5)

fp1 = Flight2D(gi, gf, bezierOrder = order, tspan= tspan, optParams=optS)

fp1.optParams.Wlen   = 0
fp1.optParams.Wcurv  = 0
fp1.optParams.Wkdev  = 1
fp1.optParams.Wspdev = 0
fp1.optParams.Wagree = 0


tic = perf_counter()
fp1.optimizeBezierPath()
toc = perf_counter()

t = np.linspace(startTime, endTime, 100)


print("Path optimization took: ", toc-tic, " seconds")

fp1.setDynConstraints(vMin, vMax, maxG)
print(fp1.timePolyCoeffs)
#fp1.optimizeTimePoly()

fp1.plotControlPoints()
fp1.plotCurve()
plt.title("Curve")

print("Time optimization took: ", toc-tic, " seconds")
x = fp1.evalPos(t)
plt.figure()
plt.title("Speed")
v = fp1.evalVel(t)
speed = np.linalg.norm(v, 2, 0)
print("Starting Speed: ", speed[0])
print("Ending Speed: ", speed[-1])

plt.plot(t, speed, 'r-')
plt.legend(['Optimized', 'Un-optimized'])
plt.show()