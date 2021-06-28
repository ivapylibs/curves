from Curves import Flight
import numpy as np
import Curves
import Curves.Bezier as Bezier
from Lie import SE2
from matplotlib import pyplot as plt

b = Bezier(3)

start = np.array([[0],[0]])
end = np.array([[5],[5]])
p1 = np.array([[1],[3]])
p2 = np.array([[4],[5]])

points2 = np.array([[4, 1, 9, 5],
           [3, 1, 8, 5]])


b.setControlPoints(points2)
t = np.linspace(0,1,10)
x = b.eval(t)
v = b.evalJet(t)
a = b.evalJet2(t)
k = b.evalCurv(t)

print(v)
print(a)
print(k)
optS = Flight.FlightOptParams(Wkdev=1)
print(Flight.Flight.BezierCostFunction(b, optS))


start = SE2()

theta = np.pi/3
R = SE2.rotationMatrix(theta)
x = np.array([[5], [4]])

end = SE2(R=R, x=x)


c = Curves.Bezier.constructBezierPath(start, end, 3, [2,1])
b.plot()
b.plotCurve(np.linspace(0,1,100))
#plt.show()