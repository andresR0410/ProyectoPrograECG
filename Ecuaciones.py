import scipy.optimize as opt
import scipy.integrate as inte

##
import scipy.optimize as opt
import scipy.integrate as inte
import numpy as np
import math as mt
import matplotlib.pyplot as plt

pi = np.pi
P = [-0.2, -pi/3, 1.2, 0.25]
Q = [-0.05, -pi/12, -5.0, 0.1]
R = [0, 0, 30.0, 0.1]
S = [0.05, pi/12, -7.5, 0.1]
T = [0.3, pi/2, 0.75, 0.4]

def dx(t,x, y):
    alpha = 1.0 - np.sqrt((x**2.0) + (y**2.0))
    w = 2.0*np.pi/t
    return alpha*x - w*y
def dy(t,x, y):
    alpha = 1.0 - np.sqrt((x ** 2.0) + (y ** 2.0))
    w = 2.0 * np.pi / t
    return alpha*x + w*y
def dz(t,x, y, z):
    suma = 0.0
    for n in [P,Q,R,S,T]:
        A = 0.00015
        th = np.arctan2(y,x)
        z0 = A * np.sin(2.0*pi * t * 0.25)
        delThi = np.mod(th - n[1], 2.0*np.pi)
        suma += n[2] * delThi * np.exp((delThi**2.0)/2.0*(n[3]**2.0))
        zt = z - z0
    res = suma - zt
    return -res

def EulerFoward(x0, y0, z0, t0, t, h, dx, dy, dz):
    T = np.arange(t0, t + h, h)
    tam = np.size(T)
    X = np.zeros(tam)
    Y = np.zeros(tam)
    Z = np.zeros(tam)
    X[0] = x0
    Y[0] = y0
    Z[0] = z0
    for i in range(1, tam):
        X[i] = X[i-1] + h* (dx(X[i-1],T[i-1],Y[i-1]))
        Y[i] = Y[i -1] + h * (dy(X[i - 1], T[i - 1], Y[i - 1]))
        Z[i] = Z[i -1] + h * (dz(X[i - 1], Y[i - 1], Z[i-1], T[i-1]))

    plt.figure()
    plt.plot(T,Z)
    return T, Z
def EulerBack(x0, y0, z0, t0, t, h, dx, dy, dz):
    T = np.arange(t0, t + h, h)
    tam = np.size(T)
    X = np.zeros(tam)
    Y = np.zeros(tam)
    Z = np.zeros(tam)
    X[0] = x0
    Y[0] = y0
    Z[0] = z0
    for i in range(1, tam):
        X[i] = X[i] - h * (dx(X[i], Y[i], Z[i], T[i]))
        Y[i] = Y[i] - h * (dy(X[i], T[i], Y[i]))
        Z[i] = Z[i] - h * (dz(X[i], Y[i], Z[i], T[i]))
    plt.figure()
    plt.plot(T, Z)
    return T, Z
##RK2 y rk4
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
thi = [-pi/2, -pi/12, 0, pi/12, pi/2]
ai = [1.2, -5.0, 30.0, -7.5, 0.75]
bi = [0.25, 0.1, 0.1, 0.1, 0.4]
Ti = np.arange(0.0, 30.0+ (1/300), 1/300)
ti = np.random.normal(60/80, 0.05*(60/80), np.size(Ti))

def dx(t, x, y):
    alpha = 1.0 - np.sqrt((x**2.0) + (y**2.0))
    w = 2.0*np.pi/t
    return alpha*x - w*y
def dy(t,x, y):
    alpha = 1.0 - np.sqrt((x ** 2.0) + (y ** 2.0))
    w = 2.0 * np.pi / t
    return alpha*x + w*y
def dz(t, x, y, z):
    suma = 0.0
    for n in range(len(ai)):
        A = 0.00015
        th = np.arctan2(y,x)
        z0 = A * np.sin(2.0*pi * t * 0.25)
        delThi = np.fmod(th - thi[n], 2.0*np.pi)
        suma += -((ai[n] * delThi * np.exp(-((delThi**2.0)/(2.0*(bi[n]**2.0)))) )-(z-z0))
    return suma


y1rk2= np.zeros(len(ti))
y2rk2= np.zeros(len(ti))
y3rk2= np.zeros(len(ti))

X0= 1
Y0= 0
Z0= 0.04
y1rk2[0]= X0
y2rk2[0]= Y0
y3rk2[0]= Z0
h= 1/300

for i in range (1,len(ti)):
    k11= dx(ti[i], y1rk2[i-1], y2rk2[i-1])
    k21= dy(ti[i], y1rk2[i-1], y2rk2[i-1])
    k31= dz(Ti[i], y1rk2[i-1], y2rk2[i-1], y3rk2[i-1])

    k12= dx(ti[i]+h/2, y2rk2[i - 1] + k11 * h/2, y2rk2[i - 1] + h/2 * k21)
    k22= dy(ti[i]+h/2, y2rk2[i - 1] + k11 * h/2, y2rk2[i - 1] + h/2 * k21)
    k32= dz(Ti[i]+h/2, y2rk2[i - 1] + k11 * h/2, y2rk2[i - 1] + h/2 * k21, y3rk2[i-1] + k31*h/2)

    y1rk2[i]= y1rk2[i-1] + (h/6)* (k11 + k12 )
    y2rk2[i]= y2rk2[i-1] + (h/6)* (k21 + k22 )
    y3rk2[i] = y3rk2[i-1] + (h / 6) * (k31 + k32)

#ploteamos:
"""fig= plt.plot()
fig.add_subplot(111).plot(th, y3rk2, "red")
fig.title("ECG")
rk2plot= FigureCanvasTkAgg(fig, master=root)
rk2plot.draw()
rk2plot.get_tk_widget().place(x=50,y=50)"""
plt.plot(Ti, y3rk2,"red")
plt.show()
plt.title("ecg")
##
y1rk4= np.zeros(len(ti))
y2rk4= np.zeros(len(ti))
y3rk4= np.zeros(len(ti))


Y0= 0
X0=0
Z0=0
y1rk4[0]= X0
y2rk4[0]= Y0
y3rk4[0]= Z0
h= 1/300

for i in range (0,len(ti)):
    k11_= dx(ti[i-1], y1rk2[i-1], y2rk2[i-1])
    k21_= dy(ti[i-1], y1rk2[i-1], y2rk2[i-1])
    k31_= dz(Ti[i-1], y1rk2[i-1], y2rk2[i-1], y3rk2[i-1])

    k12_= dx(ti[i - 1]+h/2, y2rk2[i - 1] + k11_ * h/2, y2rk2[i - 1] + h/2 * k21_)
    k22_= dy(ti[i - 1]+h/2, y2rk2[i - 1] + k11_ * h/2, y2rk2[i - 1] + h/2 * k21_)
    k32_= dz(Ti[i - 1]+h/2, y2rk2[i - 1] + k11_ * h/2, y2rk2[i - 1] + h/2 * k21_, y3rk2[i-1] + k31_*h/2)

    k13_=  dx(ti[i - 1]+h/2, y2rk2[i - 1] + k12_ * h/2, y2rk2[i - 1] + h/2 * k22_)
    k23_=  dy(ti[i - 1]+h/2, y2rk2[i - 1] + k12_ * h/2, y2rk2[i - 1] + h/2 * k22_)
    k33_=  dz(Ti[i - 1]+h/2, y2rk2[i - 1] + k12_ * h/2, y2rk2[i - 1] + h/2 * k22_, y3rk4[i-1] + h/2 * k32_)

    k14_=  dx(ti[i - 1]+h, y2rk2[i - 1] + k13_ * h, y2rk2[i - 1] + h * k23_)
    k24_=  dy(ti[i - 1]+h, y2rk2[i - 1] + k13_ * h, y2rk2[i - 1] + h * k23_)
    k34_=  dz(Ti[i - 1]+h, y2rk2[i - 1] + k13_ * h, y2rk2[i - 1] + h* k23_, y3rk4[i-1]+ h*k33_)


    y1rk4[i]= y1rk2[i-1] + (h/6)* (k11_ + 2*k12_ + 2*k13_+ k14_ )
    y2rk4[i]= y2rk2[i-1] + (h/6)* (k21_ + 2*k22_ +2*k23_+ k24_ )
    y3rk4[i] = y3rk2[i - 1] + (h / 6) * (k31_ + 2*k32_ + 2*k33_ + k34_)

plt.plot(Ti, y3rk4, "black")
plt.show()
plt.title("ecg")


#ploteamos
"""plt.plot
plt.plot(th, y3rk4, "red")
plt.show()
plt.title("ecg")"""


"""fig.add_subplot(112).plot(th, y3rk4, "red")
fig.title("ECG")
rk4plot= FigureCanvasTkAgg(fig, master= window)
rk4plot.draw()
rk4plot.get_tk_widget().place(x=50,y=50)"""
