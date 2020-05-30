"""Interacción con usuario, pide parámetros para la generación de la señal del ECG, llama las funciones de WaveGenerator
y retorna el resultado."""
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
"""import WaveGenerator as WG
import MetodosEcuaciones as ME
import PeakFinder as pkf"""

window= tk.Tk()
window.geometry("1000x600")
window.config(cursor="arrow")

pf= tk.Label(
	text= "Bienvenido al generador de señales de EGG",
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

botonsalida = tk.Button(
    master= window,
    text="salir",
    width=3,
    bg="grey",
    fg="black",
    command= salir
)
botonsalida.place(x=0, y=0)

#FRAME DE LOS PARÁMETROS
parametros= tk.Frame(master=window)
parametros.place(x=700,y=30)
parametros.config(bg="white", width=250, height=200,highlightbackground="black",highlightthickness=2)

# FRAME DEL ECG
ECG = tk.Frame(master=window)
ECG.place(x=150, y=30)
ECG.config(bg="white", width=370, height=300,highlightbackground="black",highlightthickness=2)


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

FrCar = tk.StringVar()
NLatidos= tk.StringVar()
FrMu= tk.StringVar()
Morfo= tk.StringVar()


def obtener():
    FC= FrCar.get()
    #aca debemos usar la función
    FCRes.set(FC)

    NLa= NLatidos.get()
    # aca debemos usar la función
    Lat.set(NLa)

    FrMu1=  FrMu.get()
    # aca debemos usar la función
    FM.set(FrMu1)

    Mor= Morfo.get()
    # aca debemos usar la función
    M.set(Mor)

FCRes= tk.StringVar()
Lat= tk.StringVar()
FM= tk.StringVar()
M= tk.StringVar()

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


Morfo = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =Morfo , width = 5).place(x=170, y=150)
Morfologia = tk.Button(master=parametros, textvariable = M,text="FM= ", command = obtener, width = 3).place(x=20, y=150)
d = tk.Label(master=parametros, text="MORFOLOGÍA", fg="black", bg='orange', highlightbackground="black",highlightthickness=2).place(x=70, y=150)

#ECG
#def ECG():
    #PUES FALTA TODOJAJA
    #fig = plt.Figure(figsize=(4, 2), dpi=100)
    #t = np.arange(0,10, 0.01)
    #fig.add_subplot(111).plot(t, fun(t))     # subplot(filas, columnas, item)
    #fig.suptitle(opcion.get())

    #plt.close()
    #plt.style.use('seaborn-darkgrid')

    #Plot = FigureCanvasTkAgg(fig, master=window)
    #Plot.draw()


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
for r in range(1, 3):
    for c in range(1, 6):
        cell = tk.Entry(ab, width=5)
        cell.grid(row=r, column=c)
#Botones para elegir el método de solución
def sel():
    print('seleccion')

root = tk.Frame(master=metodos)
root.place(x=0,y=0)
root.config(width=240, height=290, bg='white')
titulo = tk.Label(master=root, text='MÉTODOS DE SOLUCIÓN', bg='white')
titulo.place(x=50, y=20)
var = tk.IntVar()
R1 = tk.Radiobutton(master=root, text="Euler hacia adelante", variable=var, value=1,
                  command=sel, bg='lightgreen')
R1.place(x=50, y=44)

R2 = tk.Radiobutton(master=root, text="Euler hacia atrás", variable=var, value=2,
                  command=sel,bg='lightgreen')
R2.place(x=50, y=88)

R3 = tk.Radiobutton(master=root, text="Euler modificado", variable=var, value=3,
                  command=sel,bg='lightgreen')
R3.place(x=50, y=132)

R4 = tk.Radiobutton(master=root, text="Runge-Kutta 2", variable=var, value=4,
                  command=sel,bg='lightgreen')
R4.place(x=50, y=176)

R5 = tk.Radiobutton(master=root, text="Runge-Kutta 4", variable=var, value=5,
                  command=sel,bg='lightgreen')
R5.place(x=50, y=220)
#Heart rate

#Importar  exportar archivos

#Importar archivo
def UploadAction(event=None):
    filename = tk.filedialog.askopenfilename()
    print('Selected:', filename)

button = tk.Button(window, text='Import file', command=UploadAction, height=3, width=9, highlightbackground='black',
                   highlightthickness=2, relief='raised', bg='lightgreen')
button.place(x=5, y=30)

# Procesar parámetros dados


window.mainloop()