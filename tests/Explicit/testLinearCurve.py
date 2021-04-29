import sys
sys.path.append('./')

import numpy as np
from Explicit import Explicit
from matplotlib import pyplot as plt

tspan = [0, 1]

def line2D(t):
    out = np.array([[2], [3]])*t
    return out

t = np.linspace(tspan[0], tspan[1], 100)

curve = Explicit(line2D, tspan)

x = curve.x(t)

plt.plot(x[0,:], x[1,:])
plt.show()