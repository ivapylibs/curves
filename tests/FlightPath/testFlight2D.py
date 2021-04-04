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
tf   = 1
order = 4


#==[2] Terminal poses.

r   = 0.5*(vMin + vMax)
the =  np.pi/3
phi = -np.pi/2


gi = SE2()
gf = SE2(x=r*np.array([[np.cos(the)],[np.sin(the)]]), R=SE2.rotationMatrix(the+phi))

optS = FlightOptParams(init=5, final=3)

fp1 = Flight2D(gi, gf, order, tf, optParams=optS)



tic = perf_counter()
fp1.optimizeBezierPath()
toc = perf_counter()

t = np.linspace(0, tf, 100)

fp1.bezier.plot()
fp1.plotCurve()

print("Optimization took: ", toc-tic, " seconds")

plt.figure()
v = fp1.bezier.evalJet(t)
plt.plot(t,np.linalg.norm(v, 2, 0))
plt.show()