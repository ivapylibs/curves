import numpy as np
from Lie import SE2
import Curves.Flight2D as Flight2D
from Curves.Flight import FlightOptParams

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

optS = FlightOptParams(init=5, final=6)

fp1 = Flight2D(gi, gf, bezierOrder = order, tspan= tspan, optParams=optS)
fp1.optimizeBezierPath()

cVec = [0, 1]

for td in np.arange(startTime + 0.5, startTime + 5, 0.5):
    print(td)
    poly = Flight2D.gen5thTimePoly(cVec, td-startTime)
    print(poly)
    fp1.timePolyCoeffs = poly
    fp1.tspan = [startTime, td]
    s, dsdt = fp1.evalTimePoly(np.arange(0,1,0.1))
    print(s)
    print(fp1.evalVel(np.linspace(startTime, td, 10)))
