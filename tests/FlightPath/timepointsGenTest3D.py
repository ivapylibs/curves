import numpy as np

import Lie.group.SE3.Homog
from Lie.tangent import Element
import Curves.Flight3D as Flight3D
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

curveType = 'carousel'

if(curveType == 'circ'):
    w0 = 1/4
    a0 = 1
    def circle(t):
        rad = 10
        if(np.isscalar(t)):
            length = 1
        else:
            length = len(t)

        return a0*np.vstack((np.sin(w0*t), (1-np.cos(w0*t)), np.ones((1,length)), w0*np.cos(w0*t), w0*np.sin(w0*t), np.zeros((1,length))))
    fCurve = circle
elif(curveType == 'spiral'):
    #theta = np.pi/4
    w0 = 1/4
    a0 = 1
    vz = 1
    def spiral(t):
        if(np.isscalar(t)):
            length = 1
        else:
            length = len(t)

        return a0*np.vstack((np.sin(w0*t), (1-np.cos(w0*t)), vz*t, w0*np.cos(w0*t), w0*np.sin(w0*t),  vz*np.ones((1,length))))
    fCurve = spiral
elif(curveType == 'carousel'):
    #theta = np.pi/4
    w0 = 1/4
    zAmplitude = 0.1
    zw0 = 3/4
    def spiral(t):
        if(np.isscalar(t)):
            length = 1
        else:
            length = len(t)

        return np.vstack((np.sin(w0*t), (1-np.cos(w0*t)), zAmplitude*np.sin(zw0*t), w0*np.cos(w0*t), w0*np.sin(w0*t), zAmplitude*zw0*np.cos(zw0*t)))
    fCurve = spiral

#==[3] Curve Breakdown
tVec = np.arange(0, tSpan, th/nPtsBezier)
#pdb.set_trace()
#tVec = tVec[:-1]
xDesired = fCurve(tVec)
xActual = np.zeros((6,len(tVec)))

fp = Flight3D(bezierOrder = bezierOrder)
fp.optParams.Wlen   = 0
fp.optParams.Wcurv  = 0
fp.optParams.Wkdev  = 100
fp.optParams.Wspdev = 1
fp.optParams.Wagree = 0
fp.optParams.doTimeWarp = False

def vec2Tangent(x):
    print(x)
    pos = x.copy()[0:3]
    pos[-1] = 0
    #pdb.set_trace()
    center = np.array([0,1,0]).reshape((3,1))
    r = center - pos # Radially directed 
    unitDir = x[3:]/np.linalg.norm(x[3:])
    z = r/np.linalg.norm(r)
    y = np.cross(np.squeeze(unitDir), np.squeeze(z)).reshape((3,1))

    R = np.hstack((unitDir, y, z))

    g = Lie.group.SE3.Homog(x=x[0:3], R=R)
    return Element(g, x[3:])

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
    #fp.bezier.plot()

xDesired = fCurve(tVec)
plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot(xDesired[0,:], xDesired[1,:], xDesired[2,:], '--')
ax.plot(xActual[0,:], xActual[1,:], xActual[2,:])
#plt.gca().set_aspect('equal')

plt.figure(2)
plt.plot(tVec, xDesired[3:,:].T, '-')
plt.plot(tVec, xActual[3:,:].T, '--')
#plt.gca().set_aspect('equal')
plt.show()