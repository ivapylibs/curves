import numpy as np
from .Flight import Flight, FlightOptParams
import Lie.group.SE2.Homog
from Lie.tangent import Element
from matplotlib import pyplot as plt
import pdb

class Flight2D(Flight):
    def __init__(self, startPose=Lie.group.SE2.Homog(), endPose=Lie.group.SE2.Homog(), tspan = [0, 1], bezierOrder=3, optParams=FlightOptParams()):
        super().__init__(startPose, endPose, tspan, bezierOrder, optParams)
        self.dimension = 2
        self.spec.vec2state = lambda x: x[0:-2, :]
    
    def plotCurve(self):
        t = np.linspace(self.tspan[0], self.tspan[1],100)
        x = (self.evalPos(t)).T
        x = x.T
        plt.plot(x[0,:] ,x[1,:])
    
    def generate(self, t0, x0, t1, x1):
        self.tspan = [t0, t1]
        self.duration = t1 - t0
        #pdb.set_trace()
        if(isinstance(x0, Lie.group.SE2.Homog)):
            self.startPose = x0
            self.endPose = x1
        elif(isinstance(x0, Element)):
            self.startPose = x0.base()
            self.endPose = x1.base()
            self.optParams.init = np.linalg.norm(x0.fiber())
            self.optParams.final = np.linalg.norm(x1.fiber())
        elif(len(x0) == 4):
            startX = x0[0:2]
            endX = x1[0:2]
            startR = Lie.group.SE2.Homog.rotationMatrix(np.arctan2(x0[3], x0[2]))
            endR = Lie.group.SE2.Homog.rotationMatrix(np.arctan2(x1[3], x1[2]))

            self.startPose = Lie.group.SE2.Homog(x=startX, R=startR)
            self.endPose = Lie.group.SE2.Homog(x=endX, R=endR)
            self.optParams.init = np.linalg.norm(x0[2:])
            self.optParams.final = np.linalg.norm(x1[2:])

        self.optimizeBezierPath()
        self.timePolyCoeffs = np.array([0, 0, 0, 0, 1/(t1-t0), 0])
        #self.optimizeTimePoly()