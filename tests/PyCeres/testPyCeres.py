pyceres_location="/home/varun/ceres-bin/lib/" # assuming library was built along with ceres
# and cmake directory is called ceres-bin
import sys
sys.path.insert(0, pyceres_location)

import PyCeres # Import the Python Bindings
import numpy as np
from jax import grad

class myCostFunction(PyCeres.CostFunction):
    def __init__(self):
        super().__init__()

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        x=parameters[0][0]
        costfn = lambda x: 10-x

        residuals[0] = costfn(x)
        if(jacobians != None):
            jacobians[0][0]=grad(costfn)(x)
        
        return True

# The variable to solve for with its initial value.
initial_x=5.0
x=np.array([initial_x]) # Requires the variable to be in a numpy array

# Here we create the problem as in normal Ceres
problem=PyCeres.Problem()

# Creates the CostFunction. This example uses a C++ wrapped function which 
# returns the Autodiffed cost function used in the C++ example
cost_function=myCostFunction()

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
print( "x : " + str(initial_x) + " -> " + str(x) + "\n")