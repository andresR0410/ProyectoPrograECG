"""Interacción con usuario, pide parámetros para la generación de la señal del ECG, llama las funciones de Ecuaciones
y retorna el resultado."""
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import struct as st
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import scipy.optimize as opt
from scipy.signal import find_peaks
import os
import sys

window= tk.Tk()
window.geometry("1000x600")
window.config(cursor="arrow")

pf= tk.Label(text= "Bienvenido al generador de señales de ECG",
	foreground="white",  # Set the text color to white
    background="green")  # Set the background color to black)
pf.pack(fill=tk.X, side=tk.TOP)

def salir():
    caja = messagebox.askquestion('salir de la aplicación', '¿Está seguro que desea cerrar la aplicación?',
                                       icon='warning')
    if caja == 'yes':
        window.destroy()
    else:
        messagebox.showinfo('Retornar', 'Será retornado a la aplicación')
photo_salir = tk.PhotoImage(file = "Boton-salir.png")
photoimage_salir = photo_salir.subsample(20, 20)
botonsalida = tk.Button(master= window, image=photoimage_salir, command= salir, padx=True, pady=True, bg='red')
botonsalida.place(x=5, y=0)

corazon = Image.open('Cora.jpg')
cora_resized = corazon.resize((100,100))
cora = ImageTk.PhotoImage(cora_resized)
coraLabel = tk.Label(window, image=cora).place(x=580, y=30)

#FRAME DE LOS PARÁMETROS
parametros= tk.Frame(master=window)
parametros.place(x=700,y=30)
parametros.config(bg="white", width=250, height=200,highlightbackground="black",highlightthickness=2)

# FRAME DEL ECG
ECG = tk.Frame(master=window)
ECG.place(x=110, y=30)
ECG.config(bg="white", width=450, height=300,highlightbackground="black",highlightthickness=2)
labelECG= tk.Label(master= ECG, font= ("Helvetica", 10), text= "SEÑAL DE ECG",highlightbackground='black', highlightthickness=2,fg="black", bg='orange').place(x=170, y=5)

#FRAME PUNTOS ai bi
puntos= tk.Frame(master=window)
puntos.place(x=150, y=350)
puntos.config(bg="lightblue", width=370, height=140,highlightbackground="black",highlightthickness=2)

#FRAME MÉTODOS SOLUCION
metodos= tk.Frame(master= window)
metodos.place(x=700,y=250)
metodos.config(bg="white", width=250, height=300,highlightbackground="black",highlightthickness=2)

"""PARAMETROS PARA GRÁFICA EGG:
- Frecuencia cardíaca media
- Número de latidos
- Frecuencia de muestreo
- Morfología de la forma de onda """
tit= tk.Label(master= parametros, text= "PARÁMETROS", fg="black", bg='white', highlightbackground='black', highlightthickness=2).place(x=80, y=5)
FC = tk.IntVar()
FC.set(80)
Lat = tk.IntVar()
Lat.set(30)
FM = tk.IntVar()
FM.set(300)
FR = tk.DoubleVar()
FR.set(0.02)
FrecCar = tk.Entry(master=parametros, textvariable=FC,width = 5).place(x=50, y=30)
tk.Label(parametros, text='Frecuencia Cardiaca', bg='orange').place(x=110,y=30)
Latidos = tk.Entry(master=parametros, textvariable=Lat,width = 5).place(x=50, y=70)
tk.Label(parametros, text='Latidos', bg='orange').place(x=110,y=70)
FrecMuest = tk.Entry(master=parametros, textvariable=FM, width=5).place(x=50, y=110)
tk.Label(parametros, text='Frecuencia Muestreo', bg='orange').place(x=110,y=110)
FacRuido = tk.Entry(master=parametros, textvariable=FR,width = 5).place(x=50, y=150)
tk.Label(parametros, text='Factor Ruido', bg='orange').place(x=110,y=150)
#Tabla ai, bi
ab = tk.Frame(master=puntos)
ab.place(x=5,y=5)
ab.config(width=180, height=100, bd=3,bg='lightblue')
emptyFrame = tk.Label(master=ab, width=7, height=2, bg='lightblue').grid(row=0,column=0)
ai = tk.Label(master=ab, text='ai',  width=7, height=2,highlightbackground="black", highlightthickness=2, relief='raised').grid(row=1,column=0)
bi= tk.Label(master=ab,text='bi',  width=7, height=2,highlightbackground="black", highlightthickness=2,relief='raised').grid(row=2,column=0)
PFrame = tk.Label(master=ab, text='P', width=7, height=2, highlightbackground="black", highlightthickness=2,relief='raised').grid(row=0,column=1)
QFrame = tk.Label(master=ab,text='Q', width=7, height=2,highlightbackground="black", highlightthickness=2,relief='raised').grid(row=0,column=2)
RFrame = tk.Label(master=ab,text='R', width=7, height=2,highlightbackground="black", highlightthickness=2,relief='raised').grid(row=0,column=3)
SFrame = tk.Label(master=ab,text='S', width=7, height=2,highlightbackground="black", highlightthickness=2,relief='raised').grid(row=0,column=4)
TFrame = tk.Label(master=ab,text='T', width=7, height=2,highlightbackground="black", highlightthickness=2,relief='raised').grid(row=0,column=5)
#Se crean los espacios para que el usuario ingrese los datos
aip = tk.DoubleVar()
aip.set(1.2)
aiq = tk.DoubleVar()
aiq.set(-5.0)
air = tk.DoubleVar()
air.set(30.0)
ais = tk.DoubleVar()
ais.set(-7.5)
ait = tk.DoubleVar()
ait.set(0.75)
bip = tk.DoubleVar()
bip.set(0.25)
biq= tk.DoubleVar()
biq.set(0.1)
bir = tk.DoubleVar()
bir.set(0.1)
bis = tk.DoubleVar()
bis.set(0.1)
bit = tk.DoubleVar()
bit.set(0.4)
def obtener():
    ai_p = float(aip.get())
    ai_q = float(aiq.get())
    ai_r = float(air.get())
    ai_s = float(ais.get())
    ai_t = float(ait.get())
    bi_p = float(bip.get())
    bi_q = float(biq.get())
    bi_r = float(bir.get())
    bi_s = float(bis.get())
    bi_t = float(bit.get())
    ai = [ai_p,ai_q, ai_r, ai_s, ai_t]
    bi = [bi_p,bi_q, bi_r, bi_s, bi_t]
    FrecCard=float(FC.get())
    FrecMuestr = float(FM.get())
    LatPM = float(Lat.get())
    FactRui = float(FR.get())
    parametros_val = [FrecCard, LatPM, FrecMuestr, FactRui]
    return ai, bi, parametros_val
aiP = tk.Entry(ab,width=5, textvariable=aip).grid(row=1, column=1)
aiQ = tk.Entry(ab,width=5,textvariable=aiq).grid(row=1, column=2)
aiR = tk.Entry(ab, width=5,textvariable=air).grid(row=1, column=3)
aiS = tk.Entry(ab,width=5,textvariable=ais).grid(row=1, column=4)
aiT = tk.Entry(ab,width=5,textvariable=ait).grid(row=1, column=5)
biP = tk.Entry(ab, width=5,textvariable=bip).grid(row=2, column=1)
biQ = tk.Entry(ab, width=5,textvariable=biq).grid(row=2, column=2)
biR = tk.Entry(ab, width=5, textvariable=bir).grid(row=2, column=3)
biS = tk.Entry(ab, width=5,textvariable=bis).grid(row=2, column=4)
biT = tk.Entry(ab, width=5,textvariable=bit).grid(row=2, column=5)
#Botones para elegir el método de solución
root = tk.Frame(master=metodos)
root.place(x=0,y=0)
root.config(width=240, height=290, bg='white')
titulo = tk.Label(master=root, text='MÉTODOS DE SOLUCIÓN', bg='white')
titulo.place(x=50, y=20)

EuAd= tk.BooleanVar()
EuAt=tk.BooleanVar()
EuMod=tk.BooleanVar()
RK2val=tk.BooleanVar()
RK4val=tk.BooleanVar()
#Condiciones a usar en los métodos e implementación de los mismos
pi = np.pi
thi = [-pi / 3, -pi / 12, 0, pi / 12, pi / 2]
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
    ai = obtener()[0]
    bi = obtener()[1]
    z0 = 0.00015 * np.sin(2 * np.pi * 0.25 * t)
    for i in range(len(ai)):
        ECG += -(ai[i] * (np.fmod(np.arctan2(y, x) - thi[i], 2 * np.pi)) * np.exp(
            -((np.fmod(np.arctan2(y, x) - thi[i], 2 * np.pi)) ** 2) / (2 * (bi[i] ** 2))))
    return ECG - (z - z0)
def EulerForward(x0, y0, z0, h, Ti, ti):
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

def EulerBackRoot(yt2, t2, xt1, yt1, zt1, h, ti):
    return [xt1 + h * dx(ti, yt2[0], yt2[1]) - yt2[0],
            yt1 + h * dy(ti, yt2[0], yt2[1]) - yt2[1],
            zt1 + h * dz(t2, yt2[0], yt2[1], yt2[2]) - yt2[2]]
def EulerBack(Ti, ti, h):
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
                                 (Ti[i], X[i-1], Y[i-1], Z[i-1], h, ti[i]))
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
#funcion de encontrar picos
#Heart rate
"""que arroja la función de frecuencia cardiaca. Esta función debe recibir como parámetro un
vector asociado a un registro ECG que le permita identificar los picos de las ondas R de una
señal. Consideraciones para crear función:"""
HR = tk.BooleanVar()
def findHR():
    if HR.get():
        parametros_val = obtener()[2]
        fm = float(parametros_val[2])
        f = float(parametros_val[0])
        LPM = float(parametros_val[1])
        h = 1 / fm
        Ti = np.arange(0.0, LPM + h, h)
        ti = np.random.normal(60 / f, 0.05 * (60 / f), len(Ti))
        X = RK4(ti, Ti, h)
        time= np.arange(np.size(X))/ fm
        peaks, _ = find_peaks(X, height=0.5, width=5)  # para encontrar solo las ondas R, cada latido
        time_ecg = time[peaks]
        # distancia entre picos
        taco = np.diff(time_ecg)  # la diferencia en el tiempo
        tacobpm = taco / 60  # paso de segundos a minutos
        # la frecuencia se da:
        res = np.mean(tacobpm) #la media del taco de BPM
    else:
        res=''
    return res
# puntos iniciales
def plotear_metodos():
    fig = Figure(figsize=(5, 3), dpi=80)
    parametros_val = obtener()[2]
    fm = float(parametros_val[2])
    f = float(parametros_val[0])
    LPM = float(parametros_val[1])
    FR = float(parametros_val[-1])
    h = 1 / fm
    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.04
    Ti = np.arange(0.0, LPM + h, h)
    ti = np.random.normal(60 / f, 0.05 * (60 / f), len(Ti))
    if EuAt.get():
        ZEB = EulerBack(Ti, ti,h) + np.random.normal(FR, 0.05 * FR, len(Ti))
        fig.add_subplot(111).plot(Ti,ZEB)
    if EuAd.get():
        ZEF = EulerForward(X0, Y0, Z0, h,Ti, ti) + np.random.normal(FR, 0.05 * FR, len(Ti))
        fig.add_subplot(111).plot(Ti,ZEF)
    if EuMod.get():
        ZEM = EulerMod(X0, Y0, Z0, h, Ti, ti) + np.random.normal(FR, 0.05 * FR, len(Ti))
        fig.add_subplot(111).plot(Ti,ZEM)
    if RK2val.get():
        ZRK2 = RK2(ti, Ti, h) + np.random.normal(FR, 0.05 * FR, len(Ti))
        fig.add_subplot(111).plot(Ti,ZRK2)
    if RK4val.get():
        ZRK4 = RK4(ti, Ti, h) + np.random.normal(FR, 0.05 * FR, len(Ti))
        fig.add_subplot(111).plot(Ti,ZRK4)
    canvas = FigureCanvasTkAgg(fig, ECG)
    canvas.draw()
    canvas.get_tk_widget().place(x=10, y=30)
#Encontrar picos para hallar frecuencia cardíaca desde el ECG
HRbutShow = tk.Label(master=window, height=1, width=4, highlightbackground='black',
                     highlightthickness=2, bg="grey", textvariable=findHR()).place(x=23, y=260)

HRbutton = tk.Checkbutton(master=window, height=3, width=9, highlightbackground='black', command=findHR,
                       highlightthickness=2, bg="orange", text="Hallar HR", variable=HR,
                       onvalue=True, offvalue=False).place(x=5, y=200)

R1 = tk.Checkbutton(master=root, text="Euler hacia adelante", command= plotear_metodos,bg='lightgreen',
                    onvalue=True, offvalue=False, variable=EuAd)
R1.place(x=50, y=44)

R2 = tk.Checkbutton(master=root, text="Euler hacia atrás",command=plotear_metodos, bg='lightgreen',
                    onvalue=True, offvalue=False, variable=EuAt)
R2.place(x=50, y=88)

R3 = tk.Checkbutton(master=root, text="Euler modificado", command=plotear_metodos, bg='lightgreen',
                    onvalue=True, offvalue=False, variable=EuMod)
R3.place(x=50, y=132)

R4 = tk.Checkbutton(master=root, text="Runge-Kutta 2", command=plotear_metodos, bg='lightgreen',
                    onvalue=True, offvalue=False, variable=RK2val)
R4.place(x=50, y=176)

R5 = tk.Checkbutton(master=root, text="Runge-Kutta 4", command=plotear_metodos, bg='lightgreen',
                    onvalue=True, offvalue=False, variable=RK4val)
R5.place(x=50, y=220)

# IMPORTAR Y EXPORTAR:
#INSTRUCCIÓN:
"""El usuario podrá exportar y cargar los datos obtenidos en un archivo binario, que
contendrá los datos de la gráfica y un encabezado en formato texto (txt) con los parámetros
de configuración del modelo."""
#Importar archivo

def UploadAction(event=None):
    fig = Figure(figsize=(5, 3), dpi=80)
    filenametxt = tk.filedialog.askopenfilename()

    filenamebinDatosZ = tk.filedialog.askopenfilename()
    filenamebinDatosT = tk.filedialog.askopenfilename()

    filesize = os.path.getsize(filenamebinDatosZ)
    filesize1 = os.path.getsize(filenametxt)
    filesize2 = os.path.getsize( filenamebinDatosT)

    DatosZ= 0
    Tiempo= 0
    parametros= []
    if filesize == 0 and filesize2==0:
        print("Los datos de Z o Tiempo están vacíos is empty " )
    else:
        Read_Y = filenamebinDatosZ.read()
        filenamebinDatosZ.close()
        DatosZ = np.array(st.unpack('d' * int(len(Read_Y) / 8), Read_Y))
        Read_X = filenamebinDatosT
        filenamebinDatosT.close()
        Tiempo = np.array(st.unpack('d' * int(len(Read_X) / 8), Read_X))
        fig.add_subplot(111).plot(Tiempo, DatosZ)

    if filesize1 == 0:
        print("Los parámetros están vacíos" + str(filesize1))
    else:
        f = open(filenametxt, 'r')
        fr_cardiaca = f.readline(0)
        fr_muestreo = f.readline(1)
        lpm = f.readline(2)
        factor_ruido = f.readline(3)
        filenametxt.close()
        parametros = [fr_cardiaca, fr_muestreo, lpm, factor_ruido]
        FC.set(fr_cardiaca)
        Lat.set(lpm)
        FM.set(fr_muestreo)
        FR.set(factor_ruido)

    print('Selected:', filenametxt, filenamebinDatosZ, filenamebinDatosT, DatosZ, Tiempo, parametros)

#respectivo botón:
importButton = tk.Button(window, text='Importar datos', command=UploadAction, height=3, width=11, relief='raised',bg='lightgreen')
importButton.place(x=10, y=100)

def ExportAction():

    # GUARDAR PARAMETROS EN STRING

    parametros_val = obtener()[2]
    fm = float(parametros_val[2])
    f = float(parametros_val[0])
    LPM = float(parametros_val[1])
    ruido= float(parametros_val[3])
    h = 1 / fm
    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.04
    Ti = np.arange(0.0, LPM + h, h)
    ti = np.random.normal(60 / f, 0.05 * (60 / f), len(Ti))

    ZEB = EulerBack(Ti,ti,h) + np.random.normal(f, 0.05 * f, len(Ti))
    ZEF = EulerForward(X0, Y0, Z0, h, Ti, ti) + np.random.normal(f, 0.05 * f, len(Ti))
    ZEM = EulerMod(X0, Y0, Z0, h, Ti, ti) + np.random.normal(f, 0.05 * f, len(Ti))
    ZRK2 = RK2(ti, Ti, h) + np.random.normal(f, 0.05 * f, len(Ti))
    ZRK4 = RK4(ti, Ti, h) + np.random.normal(f, 0.05 * f, len(Ti))
    #Crear archivos bin y txt

    tiempo1= st.pack("d" * len(Ti), *Ti)
    Back= st.pack("d" * len(ZEB), *ZEB)
    Forward = st.pack("d"* len(ZEF),*ZEF)
    Modified = st.pack("d"* len(ZEM),*ZEM)
    Runge2 = st.pack("d"* len(ZRK2),*ZRK2)
    Runge4 = st.pack("d"* len(ZRK4),*ZRK4)


    ECG = open("DatosECG.bin", "wb")
    ECG.write(tiempo1)
    ECG.write(Back)
    ECG.write(Forward)
    ECG.write(Modified)
    ECG.write(Runge2)
    ECG.write(Runge4)


    ECG.close()

    para = open("parametrosECG.txt" , "w+")

    para.write("Frecuencia Cardiaca "+ str(f))
    para.write(", # de latidos "+ str(LPM))
    para.write(", Fr. muestreo "+ str(fm))
    para.write(", Factor de Ruido "+ str(ruido))

    para.close()

    print('Exporting:',tiempo1, Forward, Modified, Runge2, Runge4, parametros)
#Exportar archivos

#respectivo botón:
exportButton = tk.Button(window, text='Exportar datos', command=ExportAction, height=3, width=11,relief='raised', bg='lightgreen')
exportButton.place(x=10, y=40)

window.mainloop()