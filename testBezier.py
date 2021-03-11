import numpy as np
import Bezier

b = Bezier.Bezier2D(3)

start = np.array([[0],[0]])
end = np.array([[5],[5]])
p1 = np.array([[1],[3]])
p2 = np.array([[4],[5]])

points2 = np.array([[4, 1, 9, 5],
           [3, 1, -1, 5]])


b.setControlPoints(points2)
t = np.linspace(0,1,10)
x = b.eval(t)
v = b.evalJet(t)
a = b.evalJet2(t)
k = b.evalCurv(t)

print(Bezier.costFunctionCurvDev(b))
print(Bezier.costFunctionAgree(b))