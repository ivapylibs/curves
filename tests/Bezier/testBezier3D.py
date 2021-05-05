import numpy as np
import Curves.Bezier as Bezier
from Lie import SE3
from matplotlib import pyplot as plt

b = Bezier(3)

start = np.array([[0],[0]])
end = np.array([[5],[5]])
p1 = np.array([[1],[3]])
p2 = np.array([[4],[5]])

points2 = np.array([[0, 1, 9, 5],
           [0, 1, -1, 5], 
           [0, 3, 2, 5]])


b.setControlPoints(points2)
t = np.linspace(0,1,10)
x = b.eval(t)
v = b.evalJet(t)
a = b.evalJet2(t)
k = b.evalCurv(t)



start = SE3()

roll = np.pi/3
pitch = np.pi/4
yaw = np.pi/6
R = np.matmul(np.matmul(SE3.RotZ(yaw), SE3.RotY(pitch)), SE3.RotX(roll))

x = np.array([[3], [6], [2]])
end = SE3(R=R, x=x)

c = Bezier.constructBezierPath(start, end, 3, [2,1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
c.plot(ax)
c.plotCurve(np.linspace(0,1,100), ax)
plt.show()