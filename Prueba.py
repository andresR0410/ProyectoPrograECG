##
import scipy.optimize as opt
from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

fm= 300 #frecuencia muestreo
f= 100 #frecuencia cardiaca
LPM = 10 #latidos por min
h= 1/fm
pi = np.pi
thi = [-pi/3, -pi/12, 0, pi/12, pi/2]
ai = [1.2, -5.0, 30.0, -7.5, 0.75]
bi = [0.25, 0.1, 0.1, 0.1, 0.4]
# puntos iniciales
X0= 1.0
Y0= 0.0
Z0= 0.04
Ti = np.arange(0.0, LPM+h, h)
ti = np.random.normal(60/f, 0.05*(60/f), len(Ti))
def dx(ti, x, y):
    alpha = 1.0 - np.sqrt((x**2.0) + (y**2.0))
    w = 2.0*np.pi* (1/ti)
    return alpha*x - w*y
def dy(ti,x, y):
    alpha = (1.0 - np.sqrt((x ** 2.0) + (y ** 2.0)))
    w = 2.0 * np.pi * (1/ti)
    return alpha*y + w*x
def Fz(t, x, y, z, a, b, o):
    ECG = 0
    z0 = 0.00015 * np.sin(2 * np.pi * 0.25 * t)
    for i in range(len(a)):
        ECG += -(a[i] * (np.fmod(np.arctan2(y, x) - o[i], 2 * np.pi)) * np.exp(
            -((np.fmod(np.arctan2(y, x) - o[i], 2 * np.pi)) ** 2) / (2 * (b[i] ** 2))))
    return ECG - (z - z0)
def EulerFoward(x0, y0, z0, h):
    tam = np.size(Ti)
    XFor = np.zeros(tam)
    YFor = np.zeros(tam)
    ZFor = np.zeros(tam)
    XFor[0] = x0
    YFor[0] = y0
    ZFor[0] = z0
    for i in range(1, tam):
        XFor[i] = XFor[i-1] + h * (dx(ti[i],XFor[i-1],YFor[i-1]))
        YFor[i] = YFor[i - 1] + h * (dy(ti[i],XFor[i - 1], YFor[i - 1]))
        ZFor[i] = ZFor[i - 1] + h * (Fz(Ti[i-1], XFor[i-1], YFor[i-1], ZFor[i-1], ai, bi, thi))
    return ZFor

plt.plot(Ti, EulerFoward(1.0, 0.0, 0.04, h) + np.random.normal(0.03, 0.05*0.03, len(Ti)))