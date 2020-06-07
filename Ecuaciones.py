##

import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np

fm= 300 #frecuencia muestreo
f= 80 #frecuencia cardiaca
LPM = 30 #latidos por min
ai = [1.2, -5.0, 30.0, -7.5, 0.75]
bi = [0.25, 0.1, 0.1, 0.1, 0.4]

# puntos iniciales
h= 1/fm
pi = np.pi
thi = [-pi / 3, -pi / 12, 0, pi / 12, pi / 2]

X0= 1.0
Y0= 0.0
Z0= 0.04
Ti = np.arange(0.0, LPM+ h, h)
ti = np.random.normal(60/f, 0.05*(60/f), len(Ti))

def dx(ti, x, y):
    alpha = (1.0 - np.sqrt((x**2.0) + (y**2.0)))
    w = 2.0*np.pi* (1/ti)
    return alpha*x - w*y
def dy(ti,x, y):
    alpha = (1.0 - np.sqrt((x ** 2.0) + (y ** 2.0)))
    w = 2.0 * np.pi * (1/ti)
    return alpha*y + w*x
def dz(t, x, y, z):
    ECG = 0
    z0 = 0.00015 * np.sin(2 * np.pi * 0.25 * t)
    for i in range(len(ai)):
        ECG += -(ai[i] * (np.fmod(np.arctan2(y, x) - thi[i], 2 * np.pi)) * np.exp(
            -((np.fmod(np.arctan2(y, x) - thi[i], 2 * np.pi)) ** 2) / (2 * (bi[i] ** 2))))
    return ECG - (z - z0)
def EulerFoward(x0, y0, z0, h):
    tam = np.size(Ti)
    X = np.zeros(tam)
    Y = np.zeros(tam)
    Z = np.zeros(tam)
    X[0] = x0
    Y[0] = y0
    Z[0] = z0
    for i in range(1, tam):
        X[i] = X[i-1] + h * (dx(ti[i-1],X[i-1],Y[i-1]))
        Y[i] = Y[i - 1] + h * (dy(ti[i-1],X[i - 1],  Y[i - 1]))
        Z[i] = Z[i - 1] + h * (dz(Ti[i-1],X[i - 1], Y[i-1], Z[i-1]))
    return Z

def EulerBackRoot(yt2, xt1, yt1, zt1, h, ti):
    return [xt1 + h * dx(ti, yt2[0], yt2[1]) - yt2[0],
            yt1 + h * dy(ti, yt2[0], yt2[1]) - yt2[1],
            zt1 + h * dz(Ti, yt2[0], yt2[1], yt2[2]) - yt2[2]]
def EulerBack():
    tam = np.size(Ti)
    X = np.zeros(tam)
    Y = np.zeros(tam)
    Z = np.zeros(tam)
    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.04
    X[0] = X0
    Y[0] = Y0
    Z[0] = Z0
    for i in range(1, tam):
        SolEulerBack= opt.fsolve(EulerBackRoot, np.array([X[i-1],Y[i-1], Z[i-1]]),
                                 (Ti[i], X[i-1], Y[i-1], Z[i-1], h, Ti[i]))
        X[i] = SolEulerBack[0]
        Y[i] = SolEulerBack[1]
        Z[i] = SolEulerBack[2]
    return Z

def EulerModRoot(yt3, t1, t2, y1t1, y2t1, y3t1, h, ti):
    return [y1t1 + (h/2.0) * (dx(ti, yt3[0],yt3[1]) + dx(ti, y1t1,y2t1)) -
            yt3[0], y2t1 + (h / 2.0) * (dy(ti,y1t1, y2t1) + dy(ti,yt3[0], yt3[1])) -
            yt3[1], y3t1 + (h / 2.0) * (dz(t1,y1t1,y2t1,y3t1) + dz(t2, yt3[0],yt3[1],yt3[2])) - yt3[2]]
def EulerMod(X0,Y0,Z0,h,Ti,ti):
    tam = np.size(Ti)
    XMod = np.zeros(tam)
    YMod = np.zeros(tam)
    ZMod = np.zeros(tam)
    XMod[0] = X0
    YMod[0] = Y0
    ZMod[0] = Z0
    for x in range(1, tam):
        solMod = opt.fsolve(EulerModRoot,
                            np.array([XMod[x - 1], YMod[x - 1], ZMod[x-1]]),
                            (Ti[x - 1], Ti[x], XMod[x - 1], YMod[x - 1], ZMod[x - 1], h, ti[x]),xtol= 10**-10)
        XMod[x] = solMod[0]
        YMod[x] = solMod[1]
        ZMod[x] = solMod[2]
    return ZMod
# RK2 y rk4
def RK2(ti, Ti,h1):
    y1rk2= np.zeros(len(ti))
    y2rk2= np.zeros(len(ti))
    y3rk2= np.zeros(len(ti))

    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.04

    y1rk2[0]= X0
    y2rk2[0]= Y0
    y3rk2[0]= Z0

    for i in range (1,len(ti)):
        k11= dx(ti[i], y1rk2[i-1], y2rk2[i-1])
        k21= dy(ti[i], y1rk2[i-1], y2rk2[i-1])
        k31= dz(Ti[i], y1rk2[i-1], y2rk2[i-1], y3rk2[i-1])

        k12= dx(ti[i], y1rk2[i - 1] + k11 * (h1 /2), y2rk2[i - 1] + (h1/2) * k21)
        k22= dy(ti[i], y1rk2[i - 1] + k11 * (h1/2), y2rk2[i - 1] + (h1/2) * k21)
        k32= dz(Ti[i], y1rk2[i - 1] + k11 * (h1/2), y2rk2[i - 1] + (h1/2) * k21, y3rk2[i-1] + k31*(h1/2))

        y1rk2[i]= y1rk2[i-1] + (h1/2)* (k11 + k12 )
        y2rk2[i]= y2rk2[i-1] + (h1/2)* (k21 + k22 )
        y3rk2[i] = y3rk2[i-1] + (h1/2) *(k31 + k32)

    return y3rk2

def RK4(ti, Ti,h1):
    y1rk4= np.zeros(len(ti))
    y2rk4= np.zeros(len(ti))
    y3rk4= np.zeros(len(ti))
    Y0= 0
    X0=1
    Z0=0.04
    y1rk4[0]= X0
    y2rk4[0]= Y0
    y3rk4[0]= Z0

    for i in range (1,len(ti)):
        k11_= dx(ti[i], y1rk4[i-1], y2rk4[i-1])
        k21_= dy(ti[i], y1rk4[i-1], y2rk4[i-1])
        k31_= dz(Ti[i], y1rk4[i-1], y2rk4[i-1], y3rk4[i-1])

        k12_= dx(ti[i], y1rk4[i - 1] + k11_ * h1/2, y2rk4[i - 1] + h1/2 * k21_)
        k22_= dy(ti[i], y1rk4[i - 1] + k11_ * h1/2, y2rk4[i - 1] + h1/2 * k21_)
        k32_= dz(Ti[i], y1rk4[i - 1] + k11_ * h1/2, y2rk4[i - 1] + h1/2 * k21_, y3rk4[i-1] + k31_*h1/2)

        k13_=  dx(ti[i], y1rk4[i - 1] + k12_ * h1/2, y2rk4[i - 1] + h1/2 * k22_)
        k23_=  dy(ti[i], y1rk4[i - 1] + k12_ * h1/2, y2rk4[i - 1] + h1/2 * k22_)
        k33_=  dz(Ti[i], y1rk4[i - 1] + k12_ * h1/2, y2rk4[i - 1] + h1/2 * k22_, y3rk4[i-1] + h1/2 * k32_)

        k14_=  dx(ti[i], y1rk4[i - 1] + k13_ * h1, y2rk4[i - 1] + h1 * k23_)
        k24_=  dy(ti[i], y1rk4[i - 1] + k13_ * h1, y2rk4[i - 1] + h1 * k23_)
        k34_=  dz(Ti[i], y1rk4[i - 1] + k13_ * h1, y2rk4[i - 1] + h1* k23_, y3rk4[i-1]+ h1*k33_)

        y1rk4[i]= y1rk4[i-1] + (h1/6)* (k11_ + 2*k12_ + 2*k13_+ k14_ )
        y2rk4[i]= y2rk4[i-1] + (h1/6)* (k21_ + 2*k22_ +2*k23_+ k24_ )
        y3rk4[i] = y3rk4[i - 1] + (h1 / 6) * (k31_ + 2*k32_ + 2*k33_ + k34_)

    return y3rk4

plt.figure()
plt.plot(Ti, RK2(ti,Ti,h),"red")
plt.plot(Ti, EulerBack(), "black")
plt.plot(Ti, RK4(ti,Ti,h),"blue")
plt.plot(Ti, EulerFoward(X0, Y0, Z0, h), "yellow")
plt.plot(Ti, EulerMod(X0,Y0,Z0,h,Ti,ti), "purple")
plt.legend(['RK2', "EulerBack","RK4","EulerFor", "EulerMod"])
plt.show()
plt.title("ecg")