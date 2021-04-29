import numpy as np
from Flight import Flight, FlightOptParams
from matplotlib import pyplot as plt

class Flight2D(Flight):
    def __init__(self, startPose, endPose, tspan = [0, 1], bezierOrder=3, optParams=FlightOptParams()):
        super().__init__(startPose, endPose, tspan, bezierOrder, optParams)
        self.dimension = 2
    
    def plotCurve(self):
        t = np.linspace(self.tspan[0], self.tspan[1],100)
        x = (self.evalPos(t)).T
        x = x.T
        plt.plot(x[0,:] ,x[1,:])

    def generate(self, t0, x0, t1, x1):
        self.tspan = [x0, x1]
        if(type(x0) == 'Lie.SE2.SE2'):
            self.startPose = x0
            self.endPose = x1
        elif(np.shape(x0) == (4,)):
            startX = x0[0:4:2]
            endX = x1[0:4:2]
            startR = SE2.rotationMatrix(np.arctan2(x0[3], x0[1]))
            endR = SE2.rotationMatrix(np.arctan2(x1[3], x1[1]))

            self.startPose = SE2(x=startX, R=startR)
            self.endPose = SE2(x=endX, R=endR)
        
        self.optimizeBezierPath()
        self.optimizeTimePoly()