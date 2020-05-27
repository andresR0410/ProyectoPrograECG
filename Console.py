"""Interacción con usuario, pide parámetros para la generación de la señal del ECG, llama las funciones de WaveGenerator
y retorna el resultado."""
import WaveGenerator as WG

##
import tkinter as tk


window= tk.Tk()
window.config(cursor="arrow")

pf= tk.Label(
	text= "Bienvenido a nuestro proyecto EGG"
	foreground="white",  # Set the text color to white
    background="black"  # Set the background color to blac
)
pf.pack()

def salir():
    caja = tk.messagebox.askquestion('salir de la aplicación', '¿Está seguro que desea cerrar la aplicación?',
                                       icon='warning')
    if caja == 'yes':
        window.destroy()
    else:
        tk.messagebox.showinfo('Retornar', 'Será retornado a la aplicación')

botonsalida = tk.Button(
    master= window,
    text="salir",
    width=5,
    height=5,
    bg="white",
    fg="red",
    command= salir,
)
#FRAME DE LOS PARÁMETROS
parametros= tk.Frame(master=window)
parametros.place(x=15, y=20)
parametros.config(bg="blue", width=300, height=140, relief=tk.GROOVE, bd=8)

# FRAME DEL ECG
ECG = tk.Frame(master=window).pack()
ECG = tk.Frame(master=window)
ECG.place(x=15, y=230)
ECG.config(bg="yellow", width=300, height=150, relief=tk.RIDGE, bd=8)

#FRAME PUNTOS ai bi
puntos= tk.Frame(master=window).pack()
puntos= tk.Frame(master=window)
puntos.place(x=100,y=20)
puntos.config(bg="white", width=300, height=150, relief=tk.RIDGE, bd=8)

#FRAME MÉTODOS SOLUCION
metodos= tk.Frame(master= window).pack()\
metodos= tk.Frame(master=window)
metodos.place(x=100,y=150)
metodos.config(bg="red", width=300, height=150, relief=tk.RIDGE, bd=8)



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

VFrCar = tk.Spinbox(master=parametros, from_=0, to=300, textvariable = FrCar, width = 5).place(x=150, y=40)
FC = tk.Label(master=parametros, textvariable = FCRes,text="FC= ", command = obtener, width = 3).place(x=150, y=45)

VLat = tk.Spinbox(master=parametros, from_=0, to=300, textvariable = NLatidos, width = 5).place(x=150, y=50)
Latidos = tk.Label(master=parametros, textvariable = Lat,text="Lat= ", command = obtener, width = 3).place(x=150, y=55)


VFM = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =FrMu , width = 5).place(x=150, y=60)
FMu = tk.Label(master=parametros, textvariable = FM,text="FM= ", command = obtener, width = 3).place(x=150, y=65)

Morfo = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =Morfo , width = 5).place(x=150, y=70)
Morfologia = tk.Label(master=parametros, textvariable = M,text="FM= ", command = obtener, width = 3).place(x=150, y=75)

#ECG
def ECG():
    #PUES FALTA TODOJAJA
    fig = plt.Figure(figsize=(4, 2), dpi=100)
    #t = np.arange(0,10, 0.01)
    #fig.add_subplot(111).plot(t, fun(t))     # subplot(filas, columnas, item)
    #fig.suptitle(opcion.get())

    plt.close()
    plt.style.use('seaborn-darkgrid')

    Plot = FigureCanvasTkAgg(fig, master=window)
    Plot.draw()


#PUNTOS ai bi



window.mainloop()

##

