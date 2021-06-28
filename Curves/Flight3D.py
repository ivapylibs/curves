import numpy as np
from .Flight import Flight, FlightOptParams
from Lie import SE3
from Lie.tangent import Element
from matplotlib import pyplot as plt
import pdb

class Flight3D(Flight):
    def __init__(self, startPose=SE3(), endPose=SE3(), tspan = [0, 1], bezierOrder=3, optParams=FlightOptParams()):
        super().__init__(startPose, endPose, tspan, bezierOrder, optParams)
        self.dimension = 3
    
    def plotCurve(self, axes=None):
        t = np.linspace(self.tspan[0], self.tspan[1],100)
        x = (self.evalPos(t)).T
        x = x.T
        axes.plot3D(x[0,:] ,x[1,:], x[2,:])
    
    def x(self, t):
        pos = self.evalPos(t)
        vel = self.evalVel(t)
        return np.vstack((pos[0,:], vel[0,:], pos[1,:], vel[1,:], pos[2,:], vel[2,:]))

    def generate(self, t0, x0, t1, x1):
        self.tspan = [t0, t1]
        self.duration = t1 - t0
        if(isinstance(x0, SE3)):
            self.startPose = x0
            self.endPose = x1
        elif(isinstance(x0, Element)):
            self.startPose = x0.base()
            self.endPose = x1.base()
            self.optParams.init = np.linalg.norm(x0.fiber())
            self.optParams.final = np.linalg.norm(x1.fiber())
        elif(len(x0) == 6):
            print("From Vec")
            raise NotImplementedError("Vector generation not implemented yet!")
        
        self.optimizeBezierPath()
        self.timePolyCoeffs = np.array([0, 0, 0, 0, 1/(t1- t0), 0])
        #pdb.set_trace()
        #self.optimizeTimePoly()