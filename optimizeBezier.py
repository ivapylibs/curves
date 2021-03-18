pyceres_location="/home/varun/ceres-bin/lib/" # assuming library was built along with ceres
# and cmake directory is called ceres-bin
import sys
sys.path.insert(0, pyceres_location)

import PyCeres # Import the Python Bindings
import numpy as np
import Bezier
from Lie import SE2
from jax import grad

class myCostFunction(PyCeres.CostFunction):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2])

    def Evaluate(self, parameters, residuals, jacobians):
        d = np.zeros((2,))
        d[0] = parameters[0][0]
        d[1] = parameters[0][1]

        residuals[0] = self.costFromParams(d)

        if (jacobians != None):
            gradfun = grad(self.costFromParams)
            gradient = gradfun(d)
            jacobians[0][0] = gradient[0]
            jacobians[0][1] = gradient[1]

        return True

    def costFromParams(self, d):
        b = Bezier.generateBezierParam(self.start, self.end, d)
        return Bezier.costFunctionCurvDev(b)



start = SE2()

theta = np.pi/2
R = SE2.rotationMatrix(theta)
x = np.array([[20], [15]])

end = SE2(R=R, x=x)

initial_guess = [5,5]

# The variable to solve for with its initial value.
x=np.array(initial_guess) # Requires the variable to be in a numpy array

# Here we create the problem as in normal Ceres
problem=PyCeres.Problem()

# Creates the CostFunction. This example uses a C++ wrapped function which 
# returns the Autodiffed cost function used in the C++ example
cost_function=myCostFunction(start, end)

# Add the costfunction and the parameter numpy array to the problem
problem.AddResidualBlock(cost_function,None,x) 

# Setup the solver options as in normal ceres
options=PyCeres.SolverOptions()
# Ceres enums live in PyCeres and require the enum Type
options.linear_solver_type=PyCeres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout=True
summary=PyCeres.Summary()
# Solve as you would normally
PyCeres.Solve(options,problem,summary)
print(summary.BriefReport() + " \n")
print( "x : " + str(initial_guess) + " -> " + str(x) + "\n")