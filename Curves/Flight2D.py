import numpy as np
from .Flight import Flight, FlightOptParams
from Lie import SE2
from matplotlib import pyplot as plt
import pdb

class Flight2D(Flight):
    def __init__(self, startPose=SE2(), endPose=SE2(), tspan = [0, 1], bezierOrder=3, optParams=FlightOptParams()):
        super().__init__(startPose, endPose, tspan, bezierOrder, optParams)
        self.dimension = 2
    
    def plotCurve(self):
        t = np.linspace(self.tspan[0], self.tspan[1],100)
        x = (self.evalPos(t)).T
        x = x.T
        plt.plot(x[0,:] ,x[1,:])
    
    def x(self, t):
        pos = self.evalPos(t)
        vel = self.evalVel(t)
        return np.vstack((pos[0,:], vel[0,:], pos[1,:], vel[1,:]))

    def generate(self, t0, x0, t1, x1):
        self.tspan = [t0, t1]
        self.duration = t1 - t0
        if(isinstance(x0, SE2)):
            self.startPose = x0
            self.endPose = x1
        elif(len(x0) == 4):
            print("From Vec")
            startX = x0[0:4:2]
            endX = x1[0:4:2]
            #pdb.set_trace()
            startR = SE2.rotationMatrix(np.arctan2(x0[3], x0[1]))
            endR = SE2.rotationMatrix(np.arctan2(x1[3], x1[1]))

            self.startPose = SE2(x=startX, R=startR)
            self.endPose = SE2(x=endX, R=endR)
            self.optParams.init = np.linalg.norm(x0[1:4:2])
            self.optParams.final = np.linalg.norm(x1[1:4:2])
        
        self.optimizeBezierPath()
        self.optimizeTimePoly()