import numpy as np
import Bezier
from Lie import SE2
from matplotlib import pyplot as plt

import minisam

class FlightOptParams:
    def __init__(self, dt = 0.01, Wcurv = 1, Wlen=500, Wagree = 0, Wspdev = 1, Wkdev = 1, rho = 0.001, init = None, final = None):

        self.dt = dt

        self.Wcurv  = Wcurv
        self.Wlen   = Wlen
        self.Wagree = Wagree
        self.Wspdev = Wspdev
        self.Wkdev  = Wkdev
        self.rho    = rho
        self.init = init
        self.final = final

class BezierCurveFactor(minisam.NumericalFactor):
    def __init__(self, key, start, end, order, loss, optParams=FlightOptParams()):
        minisam.NumericalFactor.__init__(self, 1, [key], loss)
        self._start = start
        self._end = end
        self._order = order
        self.lossFunction = loss
        self.optParams = optParams

    # make a deep copy
    def copy(self):
        return BezierCurveFactor(self.keys()[0], self._start, self._end, self._order, self.lossFunction, self.optParams)

    # error = Bezier cost function
    def error(self, variables):
        my_params = variables.at(self.keys()[0])
        params = np.empty((2*(self._order-3) + 2, ))
        if(self.optParams.init != None and self.optParams.final == None):
            params[0] = self.optParams.init/self._order
            params[1:] = my_params
        elif(self.optParams.init != None and self.optParams.final != None):
            params[0] = self.optParams.init/self._order
            params[1] = self.optParams.final/self._order
            params[2:] = my_params
        else:
            params = my_params
        b = Bezier.constructBezierPath(self._start, self._end, self._order, params)
        return np.array([Flight2D.BezierCostFunction(b, self.optParams)])

class TimePolyFactor(minisam.NumericalFactor):
    def __init__(self, key, curve, minSpd, maxSpd, maxGs, loss):
        minisam.NumericalFactor.__init__(self, 1, [key], loss)
        self._curve = curve
        self._minSpd = minSpd
        self._maxSpd = maxSpd
        self._maxGs = maxGs
        self._loss = loss

    # make a deep copy
    def copy(self):
        return TimePolyFactor(self.keys()[0], self._curve, self._minSpd, self._maxSpd, self._maxGs, self._loss)

    # error = Bezier cost function
    def error(self, variables):
        my_params = variables.at(self.keys()[0])
        coeffs = Flight2D.gen5thTimePoly(my_params)
        cost = Flight2D.TimeCostFunction(self._curve, coeffs, self._minSpd, self._maxSpd, self._maxGs)
        return np.array([cost])


class Flight2D:
    def __init__(self, startPose:SE2, endPose:SE2, bezierOrder=3, duration=1, optParams=FlightOptParams()):
        self.startPose = startPose
        self.endPose = endPose
        self.bezier = Bezier.Bezier(bezierOrder)
        self.duration = duration
        self.optParams = optParams
        self.timePolyCoeffs = np.array([0, 0, 0, 0, 1, 0]) # By default, polynomial does not change s input

    def constructBezierPath(self, param):
        # Changes based on dimension
        #constructBezierPath uses parameterization to define bezier curve
        pts = np.empty((2,self.bezier.order+1))

        if(self.bezier.order == 3): # 3rd Order curve with 2 free points
            pos2 = self.startPose * ( param[0] * np.array([1,0]))  
            pos3 = self.endPose   * (-param[1] * np.array([1,0]))
            pts = np.hstack((self.startPose.getTranslation(), pos2, pos3, self.endPose.getTranslation()))

        elif(self.bezier.order > 3):
            d1 = self.startPose * ( param[0] * np.array([1,0]))
            posmid = np.reshape(param[2:], (2, self.bezier.order-3))
            d2 = self.endPose   * (-param[1] * np.array([1,0]))
            pts = np.hstack((self.startPose.getTranslation(), d1, posmid, d2, self.endPose.getTranslation()))

        self.bezier.setControlPoints(pts)

    def optimizeBezierPath(self):
        if(self.bezier.order == 3 and self.optParams.init != None and self.optParams.final != None):
            self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose, 
                self.bezier.order, [self.optParams.init/self.bezier.order, self.optParams.final/self.bezier.order])
        else:
            graph=minisam.FactorGraph() 
            #loss = minisam.CauchyLoss.Cauchy(0.) # TODO: Options Struct
            loss= None
            graph.add(BezierCurveFactor(minisam.key('p', 0), self.startPose, self.endPose, self.bezier.order, loss, optParams=self.optParams))

            init_values = minisam.Variables()

            opt_param = minisam.LevenbergMarquardtOptimizerParams()
            #opt_param.verbosity_level = minisam.NonlinearOptimizerVerbosityLevel.ITERATION
            opt = minisam.LevenbergMarquardtOptimizer(opt_param)
            values = minisam.Variables()

            if(self.optParams.init != None and self.optParams.final == None):
                print("Init fixed")
                init_values.add(minisam.key('p', 0), np.ones((1+2*(self.bezier.order-3),)))

                opt.optimize(graph, init_values, values)
                d = np.array([self.optParams.init/ self.bezier.order])
                self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose,
                    self.bezier.order, np.hstack((d, values.at(minisam.key('p', 0)))))

            elif(self.optParams.init != None and self.optParams.final != None):
                print("Init and Final fixed")
                init_values.add(minisam.key('p', 0), np.ones((2*(self.bezier.order-3),)))

                opt.optimize(graph, init_values, values)
                d = np.array([self.optParams.init/ self.bezier.order, self.optParams.final / self.bezier.order])
                self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose,
                    self.bezier.order, np.hstack((d,values.at(minisam.key('p', 0)))))
            else:
                print("Fully Unconstrained")
                init_values.add(minisam.key('p', 0), np.ones((2+2*(self.bezier.order-3),)))

                opt.optimize(graph, init_values, values)
                self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose,
                    self.bezier.order, values.at(minisam.key('p', 0)))

    @staticmethod
    def gen5thTimePoly(cVec):
        poly = np.zeros((6,))
        poly[1] = 1
        poly[2:4] = cVec
        '''
        poly[4] = -3*poly[2] - 2*poly[3]
        poly[5] = 1-np.sum(poly[0:4])
        '''
        beta = np.matmul(np.linalg.inv(np.array([[4, 5], [1,1]])), np.array([[-2*cVec[0] - 3*cVec[1]], [-cVec[0] - cVec[1]]]))
        poly[4:6] = beta[:,0]
        poly = np.flip(poly)
        return poly

    def setDynConstraints(self, minSpd, maxSpd, maxGs):
        self.minSpd = minSpd
        self.maxSpd = maxSpd
        self.maxGs = maxGs

    def optimizeTimePoly(self):
        graph=minisam.FactorGraph() 
        #loss = minisam.CauchyLoss.Cauchy(0.) # TODO: Options Struct
        loss= None
        graph.add(TimePolyFactor(minisam.key('p', 0), self.bezier, self.minSpd, self.maxSpd, self.maxGs, loss))

        init_values = minisam.Variables()

        opt_param = minisam.LevenbergMarquardtOptimizerParams()
        #opt_param.verbosity_level = minisam.NonlinearOptimizerVerbosityLevel.ITERATION
        opt = minisam.LevenbergMarquardtOptimizer(opt_param)
        values = minisam.Variables()
        init_values.add(minisam.key('p', 0), np.ones((2,)))

        opt.optimize(graph, init_values, values)
        self.timePolyCoeffs = Flight2D.gen5thTimePoly(values.at(minisam.key('p', 0)))

    # In this function we use the time polynomial to "Stretch" s which is progress from 0-1
    def evalTimePoly(self, s):
        s = np.polyval(self.timePolyCoeffs, s)
        dsdt = np.polyval(np.polyder(self.timePolyCoeffs), s)
        return (s, dsdt)

    # t is real time here. The time polynomial gives progress (s) to input into bezier eval
    def evalPos(self, t):
        (s, dsdt) = self.evalTimePoly(t / self.duration)
        return self.bezier.eval(s)

    # same as evalPos but for velocity
    def evalVel(self, t):
        (s, dsdt) = self.evalTimePoly(t / self.duration)
        vs = (self.bezier.evalJet(s) * dsdt) / self.duration
        return vs

    def plotCurve(self):
        t = np.linspace(0,self.duration,100)
        x = (self.evalPos(t)).T
        x = x.T
        plt.plot(x[0,:] ,x[1,:])

    @staticmethod
    def BezierCostFunction(path:Bezier, optParams:FlightOptParams):
        # Cost function of 4 terms: total curvature, curvature variance, length, and speed variance
        t = np.arange(0, 1, optParams.dt) # time step for evaulating cost
        
        cost = 0

        v = path.evalJet(t)
        speeds = np.linalg.norm(v, 2, 0)
        k = path.evalCurv(t)

        if(optParams.Wlen > 0):
            pathLength = optParams.dt*np.nansum(speeds)
            cost += optParams.Wlen * pathLength
        
        if(optParams.Wcurv > 0):
            totalCurv = np.nansum(np.power(k,2))
            cost += optParams.Wcurv * totalCurv
        
        if(optParams.Wkdev > 0):
            curvDev = np.nanvar(k)
            cost += optParams.Wkdev * curvDev
        
        if(optParams.Wspdev > 0):
            spddev = np.nanvar(speeds)
            cost += optParams.Wspdev * spddev

        if(optParams.Wagree > 0):
            startVec = path.Q[:,1] - path.Q[:,0]
            endVec = path.Q[:,3] - path.Q[:,2]

            startAngle = np.arctan2(startVec[1], startVec[0])
            endAngle = np.arctan2(endVec[1], endVec[0])
            angles = np.linspace(startAngle, endAngle, np.shape(t)[0])
            vecs = np.vstack((np.cos(angles),np.sin(angles)))
            ramp = np.linspace(1, 0, int(len(t)/10))
            weight = np.concatenate((ramp, np.zeros((np.shape(t)[0] - 2*np.shape(ramp)[0])), np.flip(ramp)))

            agree = np.sum(angles*weight*v/(speeds + optParams.rho))
            cost += optParams.Wkdev * agree

        return cost
    
    @staticmethod
    def TimeCostFunction(path:Bezier, timePolyCoeffs, minSpd, maxSpd, maxGs):
        # Cost function of 4 terms: total curvature, curvature variance, length, and speed variance
        dt = 0.01
        t = np.arange(0, 1, dt) # time step for evaulating cost
        tau = np.polyval(timePolyCoeffs, t)
        tauPrime = np.polyval(np.polyder(timePolyCoeffs), t)

        v = path.evalJet(tau)
        speeds = np.linalg.norm(v, 2, 0) * tauPrime # Speed in real units


        cost = 0

        k = path.evalCurv(tau) # Curvature
        ac = np.power(speeds,2)*k # Centripetal Acceleration

        Wvel = 100
        Wk = 5


        if(Wvel > 0):
            cost += Wvel*np.sum(np.power(speeds - 0.5*(minSpd + maxSpd), 2))

        if(Wk > 0):
            cost += Wk * np.sum(np.power(ac, 2))

        return cost