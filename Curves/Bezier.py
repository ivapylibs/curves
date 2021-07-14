import numpy as np
from scipy.linalg import pascal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .CurveBase import CurveBase
import pdb

class Bezier(CurveBase):
    def __init__(self, order):
        super().__init__(self)
        self.order = order
        self._C = self.__compCoeffsBernstein(order)

    def setControlPoints(self, points):
        if(np.shape(points)[1] != self.order + 1):
            raise RuntimeError("Number of control points is incorrect")
        else:
            self.Q = points

    def eval(self, t):
        #if(len(np.shape(t)) == 1):
            # Do something?
        if(np.isscalar(t)):
            t = np.array(t)
        tVec = np.tile(t, (self.order+1,1)) ** np.tile((np.arange(0,self.order+1)[:, np.newaxis]), (1, t.size))
        return np.matmul(np.matmul(self.Q,self._C),tVec)


    def x(self, t):
        return self.eval(t)

    def evalJet(self,t):
        vCurve = Bezier(self.order-1)
        vPts = self.order * np.diff(self.Q, axis=1)
        #pdb.set_trace()
        vCurve.setControlPoints(vPts)
        return self.eval(t), vCurve.eval(t)

    def evalJet2(self,t):
        aCurve = Bezier(self.order-2)
        vPts = self.order*np.diff(self.Q, axis=1)
        aPts = (self.order-1)*np.diff(vPts, axis=1)
        aCurve.setControlPoints(aPts)
        x, v = self.evalJet(t)
        return x, v, aCurve.eval(t)

    def evalCurv(self, t):
        _, v, a = self.evalJet2(t)
        dimension = np.shape(self.Q)[0]

        if(dimension == 1):
            raise RuntimeError('Curvature cannot be calculated for 1D curves')
        elif(dimension == 2):
            #pdb.set_trace()
            return np.divide(np.abs(np.cross(v, a, 0, 0)), (np.linalg.norm(v,2,0)**3))
        elif(dimension == 3):
            return np.divide(np.linalg.norm(np.cross(v, a, 0, 0), 2, 1), (np.linalg.norm(v,2,0)**3))

    def plot(self, axes=None):
        dimension = len(self.Q[:,0])
        if(dimension == 2):
            plt.plot(self.Q[0,:], self.Q[1,:],'r--', marker='o')
        elif(dimension == 3):
            axes.plot3D(xs=self.Q[0,:], ys=self.Q[1,:], zs=self.Q[2,:], marker='o')

    def plotCurve(self, t, axes=None):
        dimension = len(self.Q[:,0])
        pts = self.eval(t)
        if(dimension == 2):
            plt.plot(pts[0, :], pts[1,:])
        elif(dimension == 3):
            axes.plot3D(pts[0, :], pts[1,:], pts[2,:])

    def __compCoeffsBernstein(self, n):
        if (n == 4):
            tC = np.array(
                [[1, -4,   6,  -4,  1],
                [0, 4,-12, 12,-4],
                [0, 0,  6,-12, 6],
                [0, 0,  0,  4,-4],
                [0, 0,  0,  0, 1]])
        elif (n == 3):
            tC = np.array([[1,-3, 3,-1],
                [0, 3,-6, 3],
                [0, 0, 3,-3],
                [0, 0, 0, 1]])
        elif (n == 5):
            tC = np.array([[1,-5, 10,-10,  5, -1],
                [0, 5,-20, 30,-20,  5],
                [0, 0, 10,-30, 30,-10],
                [0, 0,  0, 10,-20, 10],
                [0, 0,  0,  0,  5, -5],
                [0, 0,  0,  0,  0,  1]])
        elif (n == 2):
            tC = np.array([[1,-2, 1],
                [0, 2,-2],
                [0, 0, 1]])
        else:
            # Create Bernstein matrix for power basis for order n using the
            # Pascal matrix.
            c = (-1)**(1+np.arange(n+1)) 
            B1 = pascal(n+1,kind='upper') * c[:, np.newaxis]
            endCol = B1[:,-1]
            tC = B1 * np.tile(endCol, (n+1,1))
        return tC
    
    @staticmethod

    def constructBezierPath(startPose, endPose, order, param):
        #constructBezierPath uses parameterization to define bezier curve
        # param[0] = delta_0
        # param[1] = delta_1
        # param[2:] = points of each subsequent control point

        dimension = len(startPose.getTranslation())
        if(len(param) != 2+dimension*(order - 3)):
            raise RuntimeError("Number of control points is incorrect")
        pts = np.empty((1,1))
        b = Bezier(order)
        unit = np.zeros((dimension,))
        unit[0] = 1

        if(order == 3): # 3rd Order curve with 2 free points
            unit = np.zeros((dimension,))
            unit[0] = 1
            pos2 = startPose * ( param[0] * unit)
            pos3 = endPose   * (-param[1] * unit)
            pts = np.hstack((startPose.getTranslation(), pos2, pos3, endPose.getTranslation()))

        elif(order > 3):
            d1 = startPose * ( param[0] * unit)
            posmid = np.reshape(param[2:], (dimension, order-3))
            d2 = endPose   * (-param[1] * unit)
            pts = np.hstack((startPose.getTranslation(), d1, posmid, d2, endPose.getTranslation()))

        #pdb.set_trace()
        b.setControlPoints(pts)
        return b

