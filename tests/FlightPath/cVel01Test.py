import numpy as np
import Lie.group.SE2.Homog as SE2
from numpy.core.function_base import linspace
from Curves.Flight import FlightOptParams
from Curves import Flight2D
import matplotlib.pyplot as plt

vMin =  8
vMax = 8
maxG = 4
ts   = 1
tf = 3

bpOrd = 4

v = (vMin + vMax)/2

x0 = np.array([[0],[0]])

gi1 = SE2(x=x0, R = SE2.rotationMatrix(0))
gf1 = SE2(x=x0+np.array([[v*(tf-ts)],[0]]), R = SE2.rotationMatrix(0))

optS = FlightOptParams(init=v, final=v)

fp1 = Flight2D(bezierOrder=bpOrd, optParams=optS)
fp1.setDynConstraints(vMin, vMax, 4)
fp1.generate(ts, gi1, tf, gf1)

x1 = np.array([[4],[4]])

gi2 = SE2(x=x1, R = SE2.rotationMatrix(np.pi/4))
gf2 = gi2*SE2(x=np.array([[v*(tf-ts)], [0]]), R = SE2.rotationMatrix(0))
fp2 = Flight2D(bezierOrder=bpOrd, optParams=optS)
fp2.setDynConstraints(vMin, vMax, 4)
fp2.generate(ts, gi2, tf, gf2)

plt.figure(1)
fp1.plotCurve()
fp1.plotControlPoints()

fp2.plotCurve()
fp2.plotControlPoints()
plt.figure(2)
t = np.linspace(ts, tf, 30)
plt.plot(t, fp1.evalVel(t).T)
plt.plot(t, fp2.evalVel(t).T)
plt.show()