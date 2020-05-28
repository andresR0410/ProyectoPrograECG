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