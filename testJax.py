import numpy as np
import Bezier
from Lie import SE2
from jax import jit,partial
from jax import grad
import jax.numpy as jnp


start = SE2()

theta = np.pi/3
R = SE2.rotationMatrix(theta)
x = np.array([[5], [4]])

end = SE2(R=R, x=x)


def costFromParams(d):
    b = Bezier.generateBezierParam(start, end, d)
    return Bezier.costFunctionCurvDev(b)

gradfun = grad(costFromParams)
print(gradfun(jnp.asarray([2, 1], dtype=np.float32)))