##

import scipy.optimize as opt

import numpy as np

fm= 300 #frecuencia muestreo
f= 80 #frecuencia cardiaca
LPM = 30 #latidos por min
h= 1/fm
pi = np.pi
thi = [-pi/3, -pi/12, 0, pi/12, pi/2]
ai = [1.2, -5.0, 30.0, -7.5, 0.75]
bi = [0.25, 0.1, 0.1, 0.1, 0.4]

# puntos iniciales
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
    return alpha*x + w*y
def dz(t, x, y, z, thi):
    suma = 0.0
    for n in range(len(ai)):
        A = 0.00015
        th = np.arctan2(y,x)
        z0 = A * np.sin(2.0*pi * t * 0.25)
        deltaThi = np.fmod(th - thi[n], 2.0*np.pi)
        suma += -(ai[n] * deltaThi * np.exp(-((deltaThi)**2.0)/(2.0*bi[n]**2.0)))
    return suma - (z-z0)

def EulerFoward(x0, y0, z0, h):
    tam = np.size(Ti)
    X = np.zeros(tam)
    Y = np.zeros(tam)
    Z = np.zeros(tam)
    X[0] = x0
    Y[0] = y0
    Z[0] = z0
    for i in range(1, tam):
        X[i] = X[i-1] + h * (dx(X[i-1],ti[i-1],Y[i-1]))
        Y[i] = Y[i - 1] + h * (dy(X[i - 1], ti[i-1], Y[i - 1]))
        Z[i] = Z[i - 1] + h * (dz(X[i - 1], Y[i-1], Z[i-1], Ti[i-1]))
    return Z

def EulerBackRoot(yt2, t2, xt1, yt1, zt1, h, Ti, thi):
    return [xt1 + h * dx(Ti, yt2[0], yt2[1]) - yt2[0],
            yt1 + h * dy(Ti, yt2[0], yt2[1]) - yt2[1],
            zt1 + h * dz(t2, yt2[0], yt2[1], yt2[2], thi) - yt2[2]]
t0=0
def EulerBack(t0, t, h, thi):
    T = np.arange(t0, t + h, h)
    tam = np.size(T)
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
                                 (T[i], X[i-1], Y[i-1], Z[i-1], h, Ti[i], thi))
        X[i] = SolEulerBack[0]
        Y[i] = SolEulerBack[1]
        Z[i] = SolEulerBack[2]
    return Z


# RK2 y rk4
import numpy as np
import matplotlib.pyplot as plt



def RK2(ti, Ti, h1, thi):
    y1rk2= np.zeros(len(ti))
    y2rk2= np.zeros(len(ti))
    y3rk2= np.zeros(len(ti))

    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.04

    y1rk2[0]= X0
    y2rk2[0]= Y0
    y3rk2[0]= Z0
    h= 1/300

    for i in range (1,len(ti)):
        k11= dx(ti[i], y1rk2[i-1], y2rk2[i-1])
        k21= dy(ti[i], y1rk2[i-1], y2rk2[i-1])
        k31= dz(Ti[i], y1rk2[i-1], y2rk2[i-1], y3rk2[i-1], thi)

        k12= dx(ti[i], y2rk2[i - 1] + k11 * (h1 /2), y2rk2[i - 1] + (h1/2) * k21)
        k22= dy(ti[i], y2rk2[i - 1] + k11 * (h1/2), y2rk2[i - 1] + (h1/2) * k21)
        k32= dz(Ti[i], y2rk2[i - 1] + k11 * (h1/2), y2rk2[i - 1] + (h1/2) * k21, y3rk2[i-1] + k31*(h1/2), thi)

        y1rk2[i]= y1rk2[i-1] + (h1/6)* (k11 + k12 )
        y2rk2[i]= y2rk2[i-1] + (h1/6)* (k21 + k22 )
        y3rk2[i] = y3rk2[i-1] + (h1/6) *(k31 + k32)
        return y3rk2



def RK4(ti, Ti, h1, thi):
    y1rk4= np.zeros(len(ti))
    y2rk4= np.zeros(len(ti))
    y3rk4= np.zeros(len(ti))
    Y0= 0
    X0=1
    Z0=0.04
    y1rk4[0]= X0
    y2rk4[0]= Y0
    y3rk4[0]= Z0

    for i in range (0,len(ti)):
        k11_= dx(ti[i], y1rk4[i-1], y2rk4[i-1])
        k21_= dy(ti[i], y1rk4[i-1], y2rk4[i-1])
        k31_= dz(Ti[i], y1rk4[i-1], y2rk4[i-1], y3rk4[i-1], thi)

        k12_= dx(ti[i]+h1/2, y2rk4[i - 1] + k11_ * h1/2, y2rk4[i - 1] + h1/2 * k21_)
        k22_= dy(ti[i]+h1/2, y2rk4[i - 1] + k11_ * h1/2, y2rk4[i - 1] + h1/2 * k21_)
        k32_= dz(Ti[i]+h1/2, y2rk4[i - 1] + k11_ * h1/2, y2rk4[i - 1] + h1/2 * k21_, y3rk4[i-1] + k31_*h1/2, thi)

        k13_=  dx(ti[i]+h1/2, y2rk4[i - 1] + k12_ * h1/2, y2rk4[i - 1] + h1/2 * k22_)
        k23_=  dy(ti[i]+h1/2, y2rk4[i - 1] + k12_ * h1/2, y2rk4[i - 1] + h1/2 * k22_)
        k33_=  dz(Ti[i]+h1/2, y2rk4[i - 1] + k12_ * h1/2, y2rk4[i - 1] + h1/2 * k22_, y3rk4[i-1] + h1/2 * k32_, thi)

        k14_=  dx(ti[i]+h1, y2rk4[i - 1] + k13_ * h1, y2rk4[i - 1] + h1 * k23_)
        k24_=  dy(ti[i]+h1, y2rk4[i - 1] + k13_ * h1, y2rk4[i - 1] + h1 * k23_)
        k34_=  dz(Ti[i]+h1, y2rk4[i - 1] + k13_ * h1, y2rk4[i - 1] + h1* k23_, y3rk4[i-1]+ h1*k33_, thi)


        y1rk4[i]= y1rk4[i-1] + (h1/6)* (k11_ + 2*k12_ + 2*k13_+ k14_ )
        y2rk4[i]= y2rk4[i-1] + (h1/6)* (k21_ + 2*k22_ +2*k23_+ k24_ )
        y3rk4[i] = y3rk4[i - 1] + (h1 / 6) * (k31_ + 2*k32_ + 2*k33_ + k34_)

    return y3rk4


plt.figure()

plt.plot(Ti, RK2(ti,Ti,h, thi),"red")
plt.plot(Ti, EulerBack(t0, 30, h, thi), "black")
plt.plot(Ti, RK4(ti,Ti,h, thi),"blue")
plt.plot(Ti, EulerFoward(X0, Y0, Z0, h), "brown")

plt.show()
plt.title("ecg")



