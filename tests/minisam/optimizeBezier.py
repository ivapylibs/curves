import minisam
import numpy as np
import sys

import matplotlib.pyplot as plt

sys.path.insert(1, './') #TODO: Installation for bezier curve library
import Bezier
import Flight2D
from Lie import SE2
from time import perf_counter


# ################################## factor ####################################

class BezierCurveFactor(minisam.NumericalFactor):
    def __init__(self, key, start, end, order, loss):
        minisam.NumericalFactor.__init__(self, 1, [key], loss)
        self._start = start
        self._end = end
        self._order = order

    # make a deep copy
    def copy(self):
        return BezierCurveFactor(self.keys()[0], self._start, self._end, self._order, self.lossFunction())

    # error = Bezier cost function
    def error(self, variables):
        params = variables.at(self.keys()[0])
        b = Bezier.constructBezierPath(self._start, self._end, self._order, params)
        return np.array([Bezier.costFunctionCurvDev(b)])


#loss=minisam.CauchyLoss.Cauchy(1.0)
loss=None

start = SE2()

theta = np.pi/4
R = SE2.rotationMatrix(theta)
x = np.array([[6], [15]])

end = SE2(R=R, x=x)
order = 6

graph=minisam.FactorGraph()

graph.add(BezierCurveFactor(minisam.key('p', 0), start, end, order, loss))

init_values = minisam.Variables()
init_values.add(minisam.key('p', 0), np.ones((2+2*(order-3),)))

print("initial curve parameters :", init_values.at(minisam.key('p', 0)))

opt_param = minisam.LevenbergMarquardtOptimizerParams()
#opt_param.verbosity_level = minisam.NonlinearOptimizerVerbosityLevel.ITERATION
opt = minisam.LevenbergMarquardtOptimizer(opt_param)

values = minisam.Variables()

tic = perf_counter()
status = opt.optimize(graph, init_values, values)
toc = perf_counter()

print("opitmized curve parameters :", values.at(minisam.key('p', 0)))
print("Optimization took: ", toc-tic)

b = Bezier.constructBezierPath(start, end, order, values.at(minisam.key('p', 0)))

dt = 0.01
t = np.arange(0, 1, dt)

b.plot()
b.plotCurve(t)

plt.show()