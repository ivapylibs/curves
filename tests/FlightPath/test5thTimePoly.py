import sys
sys.path.append('./') #TODO: Installation for bezier curve library

import numpy as np
from Lie import SE2
from Flight import Flight, FlightOptParams

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

fp1 = Flight(gi, gf, bezierOrder = order, tspan= tspan, optParams=optS)

cVec = [0, 1]

for td in np.arange(0.5, 5, 0.5):
    print(td)
    print(Flight.gen5thTimePoly(cVec, td))
