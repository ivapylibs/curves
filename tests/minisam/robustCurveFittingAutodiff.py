import minisam
import numpy as np

import matplotlib.pyplot as plt

# load x-y value pairs
def loadFromFile(filename):
    data = []
    f = open(filename, "r")
    for line in f:
        arr = [float(num) for num in line.split(' ')]
        data.append(np.array(arr))
    return data


# ################################## factor ####################################

# exp curve fitting factor
class ExpCurveFittingFactor(minisam.NumericalFactor):
    # ctor
    def __init__(self, key, point, loss):
        minisam.NumericalFactor.__init__(self, 1, [key], loss)
        self.p_ = point

    # make a deep copy
    def copy(self):
        return ExpCurveFittingFactor(self.keys()[0], self.p_, self.lossFunction())

    # error = y - exp(m * x + c);
    def error(self, variables):
        params = variables.at(self.keys()[0])
        return np.array([self.p_[1] - np.exp(params[0] * self.p_[0] + params[1])])


data = loadFromFile("/home/varun/minisam/examples/data/exp_curve_fitting_data.txt")

loss=minisam.CauchyLoss.Cauchy(1.0)
#loss=None

graph=minisam.FactorGraph()

for pair in data:
    graph.add(ExpCurveFittingFactor(minisam.key('p', 0), pair, loss))

init_values = minisam.Variables()
init_values.add(minisam.key('p', 0), np.array([0,0]))

print("initial curve parameters :", init_values.at(minisam.key('p', 0)))

opt_param = minisam.LevenbergMarquardtOptimizerParams()
opt_param.verbosity_level = minisam.NonlinearOptimizerVerbosityLevel.ITERATION
opt = minisam.LevenbergMarquardtOptimizer(opt_param)

values = minisam.Variables()
status = opt.optimize(graph, init_values, values)

print("opitmized curve parameters :", values.at(minisam.key('p', 0)))