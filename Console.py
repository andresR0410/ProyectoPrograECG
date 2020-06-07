"""Interacción con usuario, pide parámetros para la generación de la señal del ECG, llama las funciones de Ecuaciones
y retorna el resultado."""
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import struct as st
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import Ecuaciones as ec
import numpy as np

window= tk.Tk()
window.geometry("1000x600")
window.config(cursor="arrow")

pf= tk.Label(
	text= "Bienvenido al generador de señales de ECG",
	foreground="white",  # Set the text color to white
    background="green"  # Set the background color to black
)
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
coraLabel = tk.Label(window, image=cora).place(x=550, y=30)

#FRAME DE LOS PARÁMETROS
parametros= tk.Frame(master=window)
parametros.place(x=700,y=30)
parametros.config(bg="white", width=250, height=200,highlightbackground="black",highlightthickness=2)

# FRAME DEL ECG
ECG = tk.Frame(master=window)
ECG.place(x=150, y=30)
ECG.config(bg="white", width=370, height=300,highlightbackground="black",highlightthickness=2)
labelECG= tk.Label(master= ECG, font= ("Helvetica", 18), text= "SEÑAL DE ECG",highlightbackground='black', highlightthickness=2,fg="black", bg='orange').place(x=100, y=5)

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
- Morfología de la forma de onda (valores de , y )."""

FrCar = tk.DoubleVar()
NLatidos= tk.DoubleVar()
FrMu= tk.DoubleVar()
FacRuidat= tk.DoubleVar()

def obtener():
    pass
    """FC= float(FrCar.get())
    #aca debemos usar la función
    float(FCRes.set(FC))
    NLa= float(NLatidos.get())
    # aca debemos usar la función
    float(Lat.set(NLa))

    FrMu1=  float(FrMu.get())
    # aca debemos usar la función
    float(FM.set(FrMu1))

    Mor= float(FacRuidat.get())
    # aca debemos usar la función
    float(M.set(Mor))
    return [FC, NLa, FrMu1,Mor]"""
FCRes= tk.DoubleVar()
Lat= tk.DoubleVar()
FM= tk.DoubleVar()
M= tk.DoubleVar()

tit= tk.Label(master= parametros, text= "PARÁMETROS", fg="black", bg='white', highlightbackground='black', highlightthickness=2).place(x=80, y=5)

VFrCar = tk.Spinbox(master=parametros, from_=0, to=300, textvariable = FrCar, width = 5).place(x=170, y=30)
FC = tk.Button(master=parametros, textvariable = FCRes,text="FC= ", command = obtener, width = 3).place(x=20, y=30)
a = tk.Label(master= parametros, text="FC", fg="black", bg='orange', highlightbackground="black",highlightthickness=2).place(x=70, y=30)


VLat = tk.Spinbox(master=parametros, from_=0, to=300, textvariable = NLatidos, width = 5).place(x=170, y=70)
Latidos = tk.Button(master=parametros, textvariable = Lat,text="Lat= ", command = obtener, width = 3).place(x=20, y=70)
b = tk.Label(master= parametros, text="LATIDOS", fg="black", bg='orange', highlightbackground="black",highlightthickness=2).place(x=70, y=70)


VFM = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =FrMu , width = 5).place(x=170, y=110)
FMu = tk.Button(master=parametros, textvariable = FM,text="FM= ", command = obtener, width = 3).place(x=20, y=110)
c = tk.Label(master= parametros, text="F. MUESTREO", fg="black", bg='orange', highlightbackground="black",highlightthickness=3).place(x=70, y=110)


FacRui = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =FacRuidat , width = 5).place(x=170, y=150)#Morfología se reifere a factor de ruido
FactorRuido = tk.Button(master=parametros, textvariable = M,text="FM= ", command = obtener, width = 3).place(x=20, y=150)
d = tk.Label(master=parametros, text="FACTOR RUIDO", fg="black", bg='orange', highlightbackground="black",highlightthickness=2).place(x=70, y=150)
parametros_val = obtener()

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
aiq = tk.DoubleVar()
air = tk.DoubleVar()
ais = tk.DoubleVar()
ait = tk.DoubleVar()
bip = tk.DoubleVar()
biq= tk.DoubleVar()
bir = tk.DoubleVar()
bis = tk.DoubleVar()
bit = tk.DoubleVar()

def obtenerABi():
    if aiP.get():
        ai_p = float(aiP.get())
    else:
        ai_p= 1.2
    if aiQ.get():
        ai_q = float(aiQ.get())
    else:
        ai_q = -5.0
    if aiR.get():
        ai_r = float(aiR.get())
    else:
        ai_r = 30.0
    if aiS.get():
        ai_s = float(aiS.get())
    else:
        ai_s = -7.5
    if aiT.get():
        ai_t = float(aiT.get())
    else:
        ai_t = 0.75
    if biP.get():
        bi_p = float(biP.get())
    else:
        bi_p = 0.25
    if biQ.get():
        bi_q = float(biQ.get())
    else:
        bi_q = 0.1
    if biR.get():
        bi_r = float(biR.get())
    else:
        bi_r = 0.1
    if biS.get():
        bi_s = float(biS.get())
    else:
        bi_s = 0.1
    if biT.get():
        bi_t = float(biT.get())
    else:
        bi_t = 0.4
    ai = [ai_p,ai_q, ai_r, ai_s, ai_t]
    bi = [bi_p,bi_q, bi_r, bi_s, bi_t]
    return ai, bi
aiP = tk.Entry(ab,width=5)
aiP.grid(row=1, column=1)

aiQ = tk.Entry(ab,width=5)
aiQ.grid(row=1, column=2)

aiR = tk.Entry(ab, width=5)
aiR.grid(row=1, column=3)

aiS = tk.Entry(ab,width=5)
aiS.grid(row=1, column=4)

aiT = tk.Entry(ab,width=5)
aiT.grid(row=1, column=5)

biP = tk.Entry(ab, width=5)
biP.grid(row=2, column=1)

biQ = tk.Entry(ab, width=5)
biQ.grid(row=2, column=2)

biR = tk.Entry(ab, width=5)
biR.grid(row=2, column=3)

biS = tk.Entry(ab, width=5)
biS.grid(row=2, column=4)

biT = tk.Entry(ab, width=5)
biT.grid(row=2, column=5)
ai_valores = obtenerABi()[0]
bi_valores = obtenerABi()[1]

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

if parametros_val:
    fm = float(parametros_val[2])
    f = float(parametros_val[0])
    LPM = float(parametros_val[1])
    FR = float(parametros_val[-1])
else:
    fm= 300 #frecuencia muestreo
    f= 80 #frecuencia cardiaca
    LPM = 30 #latidos por min
    FR = 0.02
# puntos iniciales

figure = plt.Figure(figsize=(3, 2), dpi=100)
ax = figure.add_subplot(111)
line = FigureCanvasTkAgg(figure, window)
line.get_tk_widget().place(x=190,y=100)

def plotear_metodos():
    h = 1 / fm
    X0 = 1.0
    Y0 = 0.0
    Z0 = 0.04
    Ti = np.arange(0.0, LPM + h, h)
    ti = np.random.normal(60 / f, 0.05 * (60 / f), len(Ti))
    dataEuBa = {'Time': Ti, 'Z Values': ec.EulerBack() + np.random.normal(FR, 0.05 * FR, len(Ti))}
    dfEB = DataFrame(dataEuBa, columns=['Time', 'Z Values'])
    dataEuFor = {'Time': Ti,'Z Values': ec.EulerForward(X0, Y0, Z0, h) + np.random.normal(FR, 0.05 * FR, len(Ti))}
    dfEF = DataFrame(dataEuFor, columns=['Time', 'Z Values'])
    dataEuMod = {'Time': Ti,'Z Values': ec.EulerMod(X0, Y0, Z0, h, Ti, ti) + np.random.normal(FR, 0.05 * FR, len(Ti))}
    dfEM = DataFrame(dataEuMod, columns=['Time', 'Z Values'])
    dataRK2 = {'Time': Ti, 'Z Values': ec.RK2(ti, Ti, h) + np.random.normal(FR, 0.05 * FR, len(Ti))}
    dfRK2 = DataFrame(dataRK2, columns=['Time', 'Z Values'])
    dataRK4 = {'Time': Ti, 'Z Values': ec.RK4(ti, Ti, h) + np.random.normal(FR, 0.05 * FR, len(Ti))}
    dfRK4 = DataFrame(dataRK4, columns=['Time', 'Z Values'])
    if EuAd.get():
        df = dfEF
        df.plot(kind='line', legend=True, ax=ax, color='r', marker='o', fontsize=10)
        #llamar función de euler hacia adelante
    if EuAt.get():
        df2 = dfEB
        df2.plot(kind='line', legend=True, ax=ax, color='r', marker='o', fontsize=10)
        # llamar función de euler hacia atrás
    if EuMod.get():
        df3 = dfEM
        df3.plot(kind='line', legend=True, ax=ax, color='r', marker='o', fontsize=10)
        # llamar función de euler modificado
    if RK2val.get():
        df4 = dfRK2
        df4.plot(kind='line', legend=True, ax=ax, color='r', marker='o', fontsize=10)
        # llamar función de Runge-Kutta 2
    if RK4val.get():
        df5 = dfRK4
        df5.plot(kind='line', legend=True, ax=ax, color='r', marker='o', fontsize=10)
         # llamar función de Runge-Kutta 4

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
#Heart rate
findHR=tk.BooleanVar()
HR=tk.DoubleVar()
#hallar HR función
"""Al oprimir el botón ‘HR’ (heart rate) se debe mostrar el promedio de latidos por minuto
que arroja la función de frecuencia cardiaca. Esta función debe recibir como parámetro un
vector asociado (MANUELA: ÓSEA LOS DATOS E Z? SI) a un registro ECG que le permita identificar los picos de las ondas R de una
señal. Consideraciones para crear función:"""

datosZ= """ecg[:]""" #de aquí obtenemos Z, para hallar HR desde el ECG generado

#encontrar picos para hallar frecuencia cardíaca desde el ECG
def findHR(FrMu1,datosZ):
    frecuencia_muestreo= FrMu1
    time= datosZ/ frecuencia_muestreo
    peaks, properties = find_peaks(ecg, height=0.5, width=5)  # para encontrar solo las ondas R, cada latido
    time_ecg = time[peaks]
    time_ecg = time_ecg[1:0]
    # distancia entre picos
    taco = np.diff(time_ecg)  # la diferencia en el tiempo
    tacobpm = taco / 60  # paso de segundos a minutos
    # la frecuencia se da:
    HR = np.mean(tacobpm) #la media del taco de BPM
    return HR

HRbutShow = tk.Label(master=window, height=1, width=4, highlightbackground='black',
                             highlightthickness=2, bg="grey", textvariable=findHR).place(x=23, y=260)

HRbut= tk.Checkbutton(master=window, height= 3, width=9, highlightbackground='black', command=findHR,
                   highlightthickness=2, bg= "orange", text= "Hallar HR", variable=findHR,
                      onvalue=True, offvalue=False).place(x=5,y=200)


# IMPORTAR Y EXPORTAR:
#INSTRUCCIÓN:
"""El usuario podrá exportar y cargar los datos obtenidos en un archivo binario, que
contendrá los datos de la gráfica y un encabezado en formato texto (txt) con los parámetros
de configuración del modelo."""

#Importar archivo

def UploadAction(event=None):
    filename = tk.filedialog.askopenfilename()
    filename2 = tk.filedialog.askopenfilename()

    datosX = open(filename, 'rb')
    datosY = open(filename2, 'rb')

    Read_X = datosX.read()
    Read_Y = datosY.read()


    datosX.close()
    datosY.close()

    DatosZ = np.array(st.unpack('d' * int(len(Read_X) / 8), Read_X))
    Tiempo = np.array(st.unpack('d' * int(len(Read_Y) / 8), Read_Y))

    print('Selected:', filename,filename2, DatosZ, Tiempo)
#respectivo botón:
importButton = tk.Button(window, text='Importar datos', command=UploadAction, height=3, width=11, relief='raised',bg='lightgreen')
importButton.place(x=10, y=100)

#Exportar archivos

#GUARDAR PARAMETROS EN STRING
parametros= "Frecuencia Cardiaca, # de latidos, Frecuencia Muestreo y Factor de Ruido"

def ExportAction(event=None):
    datosXpack = st.pack(tiempo, double)
    datosYpack = st.pack(datosZ, double)
    encabezado = st.pack(parametros, char)
    print('Exporting:',datosXpack, datosYpack, encabezado)
 
#respectivo botón:
exportButton = tk.Button(window, text='Exportar datos', command=ExportAction, height=3, width=11,relief='raised', bg='lightgreen')
exportButton.place(x=10, y=40)

# Procesar parámetros dados
#ECG generar el plot
"""def ECG():
    PUES FALTA TODOJAJA
    fig = plt.Figure(figsize=(4, 2), dpi=100)
    t = np.arange(0,10, 0.01)
    fig.add_subplot(111).plot(t, fun(t))     # subplot(filas, columnas, item)
    fig.suptitle(opcion.get())

    plt.close()
    plt.style.use('seaborn-darkgrid')

    Plot = FigureCanvasTkAgg(fig, master=window)
    Plot.draw()"""




window.mainloop()