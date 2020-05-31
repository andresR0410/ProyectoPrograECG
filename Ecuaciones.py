import scipy.optimize as opt
import scipy.integrate as inte
import numpy as np
import math as mt
import matplotlib.pyplot as plt

#encontrar picos para hallar frecuencia cardíaca desde el ECG
def HR(frecuencia_muestreo):
    time= """electrocardiograma""" / frecuencia_muestreo
    peaks, properties = find_peaks(ecg, height=0.5, width=5)  # para encontrar solo las ondas R, cada latido
    time_ecg = time[peaks]
    time_ecg = time_ecg[1:0]

    """plt.plot(time_ecg, ecg)
    plt.plot(peaks/frecuencia_muestreo, ecg[peaks], "oc")  # peaks son indices dónde están los picos
    plt.show()"""

    # distancia entre picos
    taco = np.diff(time_ecg)  # la diferencia en el tiempo
    tacobpm = taco / 60  # paso de segundos a minutos

    # la frecuencia se da:
    HR = np.mean(tacobpm)
    return HR

pi = np.pi
P = [-0.2, -pi/3, 1.2, 0.25]
Q = [-0.05, -pi/12, -5.0, 0.1]
R = [0, 0, 30.0, 0.1]
S = [0.05, pi/12, -7.5, 0.1]
T = [0.3, pi/2, 0.75, 0.4]

def dx(x, t, y):
    alpha = 1.0 - np.sqrt((x**2.0) + (y**2.0))
    w = 2.0*np.pi/t
    return alpha*x - w*y
def dy(x, t, y):
    alpha = 1.0 - np.sqrt((x ** 2.0) + (y ** 2.0))
    w = 2.0 * np.pi / t
    return alpha*x + w*y
def dz(x, y, z, t):
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
        X[i] = X[i-1] + h * (dx(X[i-1],T[i-1],Y[i-1]))
        Y[i] = Y[i - 1] + h * (dy(X[i - 1], T[i - 1], Y[i - 1]))
        Z[i] = Z[i - 1] + h * (dz(X[i - 1], Y[i - 1], Z[i-1], T[i-1]))
    plt.figure()
    plt.plot(T,Z)
    return T, Z
EulerFoward(1,0,0.04,0, 30, 0.01, dx, dy, dz)
