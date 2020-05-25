# Ecuaciones diferenciales ECG
import scipy.optimize as opt
import scipy.integrate as inte
import numpy as np
import math as mt

def dx(alpha, x, w, y):
    return alpha*x - w*y
def dy(alpha, x, w, y):
    return alpha*x + w*y
def dz(x, y, z, t, ai, bi, thi):
    th = mt.atan2(x/y)
    deltaT = th - thi
    z0 = 0.15*np.sin(2*np.pi*t)
    return -np.sum(ai * deltaT * np.exp((deltaT**2)/2*(bi**2) - z - z0))

def EulerBack(yt3, y1t1, y2t1, y3t1, h, ai, bi, T, thi):
    alpha = 1 - np.sqrt((yt3[0]**2) + yt3[1]**2)
    return [y1t1 + h * dx(alpha, yt3[0], 2*np.pi/T, yt3[1]) - yt3[0],
            y2t1 + h * dy(alpha, yt3[0], 2*np.pi/T, yt3[1]) - yt3[1],
            y3t1 + h * dz(yt3[2], yt3[1], yt3[2], T, ai, bi, thi) - yt3[2]]

def EulerMod(yt3, y1t1, y2t1, y3t1, h, ai, bi, T, thi):
    alpha = 1 - np.sqrt((yt3[0] ** 2) + yt3[1] ** 2)
    return [y1t1 + (h/2.0) * (dx(alpha, y1t1, 2*np.pi/T, y2t1)) + dx(alpha, yt3[0], 2*np.pi/T, yt3[1]) - yt3[0],
            y2t1 + (h/2.0) * (dy(alpha, y1t1, 2*np.pi/T, y2t1)) + dy(alpha, yt3[0], 2*np.pi/T, yt3[1]) - yt3[1],
            y3t1 + (h/2.0) * (dz(y1t1, y2t1, y3t1, T, ai, bi, thi)) + dz(yt3[0], yt3[1], yt3[2], T, ai, bi, thi) - yt3[2]]

T = [-0.2, -0.05, 0, 0.05, 0.3]
thi = [-np.pi/3, -np.pi/12, 0, np.pi/12, np.pi/2]
ai = [1.2, -5.0, 30.0, -7.5, 0.75]
bi = [0.25, 0.1, 0.1, 0.1, 0.4]
h = 0.01
x0 = 1
y0 = 0
z0 = 0.25

XEulerFor = np.zeros(len(T))
XEulerBack = np.zeros(len(T))
XEulerMod = np.zeros(len(T))
XRK2 = np.zeros(len(T))
XRK4 = np.zeros(len(T))

YEulerFor = np.zeros(len(T))
YEulerBack = np.zeros(len(T))
YEulerMod = np.zeros(len(T))
YRK2 = np.zeros(len(T))
YRK4 = np.zeros(len(T))

ZEulerFor = np.zeros(len(T))
ZEulerBack = np.zeros(len(T))
ZEulerMod = np.zeros(len(T))
ZRK2 = np.zeros(len(T))
ZRK4 = np.zeros(len(T))

XEulerFor[0] = x0
XEulerBack[0] = x0
XEulerMod[0] = x0
XRK2[0] = x0
XRK4[0] = x0

YEulerFor[0] = y0
YEulerBack[0] = y0
YEulerMod[0] = y0
YRK2[0] = y0
YRK4[0] = y0

ZEulerFor[0] = z0
ZEulerBack[0] = z0
ZEulerMod[0] = z0
ZRK2[0] = z0
ZRK4[0] = z0

for i in range(1, len(T)):
    alphaANT = 1 - np.sqrt((XEulerFor[i-1]**2) + YEulerFor[i-1]**2)
    XEulerFor[i] = XEulerFor[i-1] + h * dx(alphaANT, XEulerFor[i-1], 2*np.pi/T[i], YEulerFor[i-1])
    YEulerFor[i] = YEulerFor[i-1] + h * dy(alphaANT, XEulerFor[i-1], 2*np.pi/T[i], YEulerFor[i-1])
    ZEulerFor[i] = ZEulerFor[i - 1] + h * dz(XEulerFor[i-1], YEulerFor[i - 1], ZEulerFor[i-1], T[i], ai[i], bi[i], thi[i])

    Solback = opt.fsolve(EulerBack, np.array([XEulerBack[i-1], YEulerBack[i-1], ZEulerBack[i-1]]), (XEulerBack[i-1, YEulerBack[i-1],

    ]))

