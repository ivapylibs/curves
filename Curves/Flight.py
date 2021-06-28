import numpy as np
from Lie import SE2
from matplotlib import pyplot as plt
from .CurveBase import CurveBase
import abc
import pdb
import Curves.Bezier as Bezier

import minisam

class FlightOptParams:
    def __init__(self, dt = 0.01, Wcurv = 0, Wlen=0, Wagree = 0, Wspdev = 1, Wkdev = 0, rho = 0.001, init = None, final = None):

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
    def __init__(self, key, start, end, order, duration, loss=None, optParams=FlightOptParams()):
        minisam.NumericalFactor.__init__(self, 1, [key], loss)
        self._start = start
        self._end = end
        self._order = order
        self.lossFunction = loss
        self.optParams = optParams
        self.dimension = len(self._start.getTranslation())
        self.duration = duration

    # make a deep copy
    def copy(self):
        return BezierCurveFactor(self.keys()[0], self._start, self._end, self._order, self.duration, self.lossFunction, self.optParams)

    # error = Bezier cost function
    def error(self, variables):
        my_params = variables.at(self.keys()[0])
        params = np.empty((self.dimension*(self._order-3) + 2, ))
        if(self.optParams.init != None and self.optParams.final == None):
            params[0] = self.duration*(self.optParams.init/self._order)
            params[1:] = my_params
        elif(self.optParams.init != None and self.optParams.final != None):
            params[0] = self.duration*(self.optParams.init/self._order)
            params[1] = self.duration*(self.optParams.final/self._order)
            params[2:] = my_params
        else:
            params = my_params
        b = Bezier.constructBezierPath(self._start, self._end, self._order, params)
        return np.array([Flight.BezierCostFunction(b, self.optParams)])

class TimePolyFactor(minisam.NumericalFactor):
    def __init__(self, key, curve, minSpd, maxSpd, maxGs, loss, ts, tf):
        minisam.NumericalFactor.__init__(self, 1, [key], loss)
        self._curve = curve
        self._minSpd = minSpd
        self._maxSpd = maxSpd
        self._maxGs = maxGs
        self._loss = loss
        self._ts = ts
        self._tf = tf

    # make a deep copy
    def copy(self):
        return TimePolyFactor(self.keys()[0], self._curve, self._minSpd, self._maxSpd, self._maxGs, self._loss, ts=self._ts, tf=self._tf)

    # error = Bezier cost function
    def error(self, variables):
        my_params = variables.at(self.keys()[0])
        coeffs = Flight.gen5thTimePoly(my_params, self._tf - self._ts)
        cost = Flight.TimeCostFunction(self._curve, coeffs, self._minSpd, self._maxSpd, self._maxGs, [self._ts, self._tf])
        return np.array([cost])

class Flight(CurveBase):
    def __init__(self, startPose, endPose, tspan = [0, 1], bezierOrder=3, optParams=FlightOptParams()):
        super().__init__(tspan)
        self.startPose = startPose
        self.endPose = endPose
        self.bezier = Bezier(bezierOrder)
        self.tspan = tspan
        self.duration = tspan[1] - tspan[0]
        self.optParams = optParams
        self.timePolyCoeffs = np.array([0, 0, 0, 0, 1/(tspan[1]-tspan[0]), 0]) # By default, polynomial does not change s input
        #self.dimension = len(startPose.getTranslation())

    def constructBezierPath(self, param):
        # Changes based on dimension
        #constructBezierPath uses parameterization to define bezier curve
        pts = np.empty((self.dimension,self.bezier.order+1))
        unit = np.zeros(self.dimension,)
        unit[0] = 1

        if(self.bezier.order == 3): # 3rd Order curve with 2 free points
            pos2 = self.startPose * ( param[0] * unit)  
            pos3 = self.endPose   * (-param[1] * unit)
            pts = np.hstack((self.startPose.getTranslation(), pos2, pos3, self.endPose.getTranslation()))

        elif(self.bezier.order > 3):
            d1 = self.startPose * ( param[0] * unit)
            posmid = np.reshape(param[2:], (self.dimension, self.bezier.order-3))
            d2 = self.endPose   * (-param[1] * unit)
            pts = np.hstack((self.startPose.getTranslation(), d1, posmid, d2, self.endPose.getTranslation()))

        self.bezier.setControlPoints(pts)

    def optimizeBezierPath(self):
        if(self.bezier.order == 3 and self.optParams.init != None and self.optParams.final != None):
            self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose, 
                self.bezier.order, self.duration*np.array([self.optParams.init/self.bezier.order, self.optParams.final/self.bezier.order]))
        else:
            graph=minisam.FactorGraph() 
            #loss = minisam.CauchyLoss.Cauchy(0.) # TODO: Options Struct
            loss= None
            graph.add(BezierCurveFactor(minisam.key('p', 0), self.startPose, self.endPose, self.bezier.order, self.duration, loss, optParams=self.optParams))

            init_values = minisam.Variables()

            opt_param = minisam.LevenbergMarquardtOptimizerParams()
            #opt_param.verbosity_level = minisam.NonlinearOptimizerVerbosityLevel.ITERATION
            opt = minisam.LevenbergMarquardtOptimizer(opt_param)
            values = minisam.Variables()

            linePts = self.startPose.getTranslation() + \
                np.arange(0,1+1/(self.bezier.order),1/(self.bezier.order))*(self.endPose.getTranslation()- self.startPose.getTranslation())
            #pdb.set_trace()

            if(self.optParams.init != None and self.optParams.final == None):
                # TODO: Initial Conditions
                initialGuess = np.hstack((linePts[3:-2].reshape((1,-1)), 1))
                #init_values.add(minisam.key('p', 0), np.ones((1+self.dimension*(self.bezier.order-3),)))
                init_values.add(minisam.key('p', 0), initialGuess)

                opt.optimize(graph, init_values, values)
                d = np.array([self.optParams.init/ self.bezier.order])
                self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose,
                    self.bezier.order, np.hstack((d, values.at(minisam.key('p', 0)))))

            elif(self.optParams.init != None and self.optParams.final != None):
                print("Both Constrained")
                initialGuess = linePts[:,2:-2].reshape((1,-1))
                d =  self.duration*np.array([self.optParams.init/ self.bezier.order, self.optParams.final / self.bezier.order])
                unit = np.zeros((self.dimension,1))
                unit[0] = 1
                pos2 = self.startPose * (d[0] * unit)
                pos3 = self.endPose * (-d[1] * unit)
                v = (pos3 - pos2) / (self.bezier.order - 2)
                initialGuess = pos2 + np.multiply(np.arange(1,self.bezier.order-3+1), v)
                initialGuess = initialGuess.reshape((1,-1))
                #pdb.set_trace()
                init_values.add(minisam.key('p', 0), np.squeeze(initialGuess))
                #init_values.add(minisam.key('p', 0), np.ones((self.dimension*(self.bezier.order-3),)))

                opt.optimize(graph, init_values, values)
                self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose,
                    self.bezier.order, np.hstack((d,values.at(minisam.key('p', 0)))))
            else:
                initialGuess = np.hstack((linePts[3:-2].reshape((1,-1))))
                #init_values.add(minisam.key('p', 0), np.ones((2+self.dimension*(self.bezier.order-3),)))
                init_values.add(minisam.key('p', 0), initialGuess)

                opt.optimize(graph, init_values, values)
                self.bezier = Bezier.constructBezierPath(self.startPose, self.endPose,
                    self.bezier.order, values.at(minisam.key('p', 0)))

    @staticmethod
    def gen5thTimePoly(cVec, td):
        if(td != 0):
            b = np.array([0, 1-cVec[0]*td**2 - cVec[1]*td**3, 1/td , 1/td -2*cVec[0]*td - 3*cVec[1]*td**2])
            b = np.reshape(b, (4,1))
            A = np.array([
            [1, 0, 0, 0],
            [1, td, td**4, td**5],
            [0, 1, 0, 0],
            [0, 1, 4*td**3, 5*td**4],
            ])
            beta = np.matmul(np.linalg.inv(A), b)
            
            beta = beta.T
            beta = np.squeeze(beta)
            coeffs = np.flip(np.hstack((beta[0:2], cVec, beta[2:4])))
            return coeffs
            #self.timePolyCoeffs = fliplr(obj.timePolyCoeffs);
        else:
            return np.zeros((6,))

    def setDynConstraints(self, minSpd, maxSpd, maxGs):
        self.minSpd = minSpd
        self.maxSpd = maxSpd
        self.maxGs = maxGs

    def optimizeTimePoly(self):
        graph=minisam.FactorGraph() 
        #loss = minisam.CauchyLoss.Cauchy(0.) # TODO: Options Struct
        loss= None
        graph.add(TimePolyFactor(minisam.key('p', 0), self.bezier, self.minSpd, self.maxSpd, self.maxGs, loss, ts=self.tspan[0], tf= self.tspan[1]))

        init_values = minisam.Variables()

        opt_param = minisam.LevenbergMarquardtOptimizerParams()
        #opt_param.verbosity_level = minisam.NonlinearOptimizerVerbosityLevel.ITERATION
        opt = minisam.LevenbergMarquardtOptimizer(opt_param)
        values = minisam.Variables()
        init_values.add(minisam.key('p', 0), np.ones((2,)))

        opt.optimize(graph, init_values, values)
        self.timePolyCoeffs = Flight.gen5thTimePoly(values.at(minisam.key('p', 0)), self.duration)

    # In this function we use the time polynomial to "Stretch" s which is progress from 0-1
    def evalTimePoly(self, t):
        s = np.polyval(self.timePolyCoeffs, t)
        #pdb.set_trace()
        dsdt = np.polyval(np.polyder(self.timePolyCoeffs), t)
        return (s, dsdt)

    # t is real time here. The time polynomial gives progress (s) to input into bezier eval
    def evalPos(self, t):
        (s, dsdt) = self.evalTimePoly(t-self.tspan[0])
        return self.bezier.eval(s)

    def x(self, t):
        return self.evalPos(t)

    # same as evalPos but for velocity
    def evalVel(self, t):
        (s, dsdt) = self.evalTimePoly(t-self.tspan[0])
        vs = (self.bezier.evalJet(s) * dsdt)
        return vs

    def plotControlPoints(self, axes=None):
        self.bezier.plot(axes)

    @abc.abstractmethod
    def plotCurve(self, axes=None):
        return
    
    @staticmethod
    def BezierCostFunction(path:Bezier, optParams:FlightOptParams):
        # Cost function of 4 terms: total curvature, curvature variance, length, and speed variance
        t = np.arange(0, 1+optParams.dt, optParams.dt) # time step for evaulating cost
        
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
            curvDev = np.nanvar(k, ddof=1)
            cost += optParams.Wkdev * curvDev
        
        if(optParams.Wspdev > 0):
            spddev = np.nanvar(speeds, ddof=1)
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

        print(cost)
        #pdb.set_trace()
        return cost
    
    @staticmethod
    def TimeCostFunction(path:Bezier, timePolyCoeffs, minSpd, maxSpd, maxGs, tspan):
        # Cost function of 4 terms: total curvature, curvature variance, length, and speed variance
        dt = 0.01
        t = np.arange(0, tspan[1]-tspan[0], dt) # time step for evaulating cost
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