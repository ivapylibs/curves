import numpy as np

import Lie.group.SE2.Homog
from Lie.tangent import Element
from numpy.core.numeric import isscalar
import Curves.Flight2D as Flight2D
import matplotlib.pyplot as plt

import pdb


#==[1] Specification
#

vMin =  0
vMax = 10
maxG = 4

tSpan = 20
th = 2 # Time Horizon
nPtsBezier = 100 # number of points to evaluate bezier curve at

bezierOrder = 4

curveType = 'line'

if(curveType == 'circ'):
    w0 = 1/4
    a0 = 20
    def arc(t):
        rad = 10
        if(np.isscalar(t)):
            length = 1
        else:
            length = len(t)

        return a0*np.vstack((np.sin(w0*t), (1-np.cos(w0*t)), w0*np.cos(w0*t), w0*np.sin(w0*t)))
    fCurve = arc
elif(curveType == 'line'):
    theta = np.pi/4
    v = 8
    def line(t):
        if(np.isscalar(t)):
            length = 1
        else:
            length = len(t)

        xc = np.cos(theta)
        yc = np.sin(theta)
        out = np.vstack((v*xc*t, v*yc*t, v*xc*np.ones((1,length)), v*yc*np.ones((1,length))))
        return out
    fCurve = line
    #pdb.set_trace()

#==[3] Curve Breakdown
tVec = np.arange(0, tSpan, th/nPtsBezier)
#pdb.set_trace()
#tVec = tVec[:-1]
xDesired = fCurve(tVec)
xActual = np.zeros((4,len(tVec)))

fp = Flight2D(bezierOrder = bezierOrder)
fp.optParams.Wlen   = 0
fp.optParams.Wcurv  = 0
fp.optParams.Wkdev  = 1
fp.optParams.Wspdev = 1
fp.optParams.Wagree = 0
fp.optParams.doTimeWarp = False

def vec2Tangent(x):
    #print(x)
    s = x[0:2]
    angle = np.arctan2(x[3], x[2])
    g = Lie.group.SE2.Homog(x=s, R=Lie.group.SE2.Homog.rotationMatrix(angle))
    return Element(g, x[2:])

for t in np.arange(0, tSpan, th):
    t0 = t
    t1 = t+th
    xStart = fCurve(t0)
    xEnd = fCurve(t1)
    print(xStart)
    print(xEnd)

    gi = vec2Tangent(xStart)
    gf = vec2Tangent(xEnd)
    fp.generate(t0, gi, t1, gf)
    xActual[:, int((t0/th)*nPtsBezier):int((t1/th)*nPtsBezier)] = np.vstack( \
        (fp.evalPos(np.linspace(t0, t1, nPtsBezier)),\
        fp.evalVel(np.linspace(t0, t1, nPtsBezier))))
    #print(xActual)
    fp.bezier.plot()

xDesired = fCurve(tVec)
plt.figure(1)
plt.plot(xDesired[0,:], xDesired[1,:], '--')
plt.plot(xActual[0,:], xActual[1,:])
plt.gca().set_aspect('equal')

plt.figure(2)
plt.plot(tVec, xDesired[2:,:].T, '-')
plt.plot(tVec, xActual[2:,:].T, '--')
#plt.gca().set_aspect('equal')
plt.show()