import numpy as np
from scipy.linalg import pascal
from matplotlib import pyplot as plt

class Bezier:
    def __init__(self, order):
        self.order = order
        self.C = self.__compCoeffsBernstein(order)

    def setControlPoints(self, points):
        if(np.shape(points)[1] != self.order + 1):
            raise RuntimeError("Number of control points is incorrect")
        else:
            self.Q = points

    def eval(self, t):
        #if(len(np.shape(t)) == 1):
            # Do something?
        tVec = np.tile(t, (self.order+1,1)) ** np.tile((np.arange(0,self.order+1)[:, np.newaxis]), (1, len(t)))
        return np.matmul(np.matmul(self.Q,self.C),tVec)

    def evalJet(self,t):
        vCurve = Bezier(self.order-1)
        vPts = np.diff(self.Q, axis=1)
        vCurve.setControlPoints(vPts)
        return vCurve.eval(t)

    def evalJet2(self,t):
        aCurve = Bezier(self.order-2)
        vPts = np.diff(self.Q, axis=1)
        aPts = np.diff(vPts, axis=1)
        aCurve.setControlPoints(aPts)
        return aCurve.eval(t)

    def evalCurv(self, t):
        v = self.evalJet(t)
        a = self.evalJet2(t)
        dimension = np.shape(self.Q)[0]

        if(dimension == 1):
            raise RuntimeError('Curvature cannot be calculated for 1D curves')
        elif (dimension == 2):
            '''
            _v = np.concatenate((v,np.zeros((1,len(t)))), 0)
            _a = np.concatenate((a,np.zeros((1,len(t)))), 0)
            '''
            return np.divide(np.cross(v, a, 0, 0), (np.linalg.norm(v,2,0)**3))
        '''
        case 3 % plot control points in 3D.
        curv = vecnorm(cross(v,a),2,1) ./ (vecnorm(v,2,1).^3);
        '''

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


def costFunction(path):
    # Cost function of 2-3 terms: (max?) curvature, length, and "agreeance"
    dt = 0.01
    t = np.arange(0, 1, dt) # time step for evaulating cost

    # Cost Function Weights
    Kcurv = 1
    Klength = 500
    #Kagree = 1
    KcurvDev = 1
    Kspddev = 1

    v = path.evalJet(t)
    speeds = np.linalg.norm(v, 2, 0)
    pathLength = dt*np.nansum(speeds)

    spddev = np.nanvar(speeds)

    k = path.evalCurv(t)
    totalCurv = np.nansum(np.power(k,2))

    curvDev = np.nanvar(k)


