"""Interacción con usuario, pide parámetros para la generación de la señal del ECG, llama las funciones de WaveGenerator
y retorna el resultado."""
import WaveGenerator as WG

##
import tkinter as tk


window= tk.Tk()
window.config(cursor="arrow", width= 600, height=400)

pf= tk.Label(
	text= "Bienvenido a nuestro proyecto EGG",
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
    bg="black",
    fg="red",
    command= salir,
)
#FRAME DE LOS PARÁMETROS
parametros= tk.Frame(master=window)
parametros.place(x=400,y=30)
parametros.config(bg="blue", width=400, height=300,highlightbackground="black",highlightthickness=1)

# FRAME DEL ECG
ECG = tk.Frame(master=window)
ECG.place(x=15, y=300)
ECG.config(bg="yellow", width=400, height=300,highlightbackground="black",highlightthickness=1)

#FRAME PUNTOS ai bi
puntos= tk.Frame(master=window)
puntos.place(x=15, y=30)
puntos.config(bg="white", width=400, height=300,highlightbackground="black",highlightthickness=1)

#FRAME MÉTODOS SOLUCION
metodos= tk.Frame(master= window)
metodos.place(x=400,y=300)
metodos.config(bg="black", width=400, height=300,highlightbackground="black",highlightthickness=1)



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

tit= tk.Label(master= parametros, text= "Parámetros").place(x=100, y=5)

VFrCar = tk.Spinbox(master=parametros, from_=0, to=300, textvariable = FrCar, width = 5).place(x=200, y=25)
FC = tk.Button(master=parametros, textvariable = FCRes,text="FC= ", command = obtener, width = 3).place(x=50, y=25)
a = tk.Label(master= parametros, text="FC").place(x=100, y=25)


VLat = tk.Spinbox(master=parametros, from_=0, to=300, textvariable = NLatidos, width = 5).place(x=200, y=55)
Latidos = tk.Button(master=parametros, textvariable = Lat,text="Lat= ", command = obtener, width = 3).place(x=50, y=55)
b = tk.Label(master= parametros, text="LATIDOS").place(x=100, y=55)


VFM = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =FrMu , width = 5).place(x=200, y=85)
FMu = tk.Button(master=parametros, textvariable = FM,text="FM= ", command = obtener, width = 3).place(x=50, y=85)
c = tk.Label(master= parametros, text="F. MUESTREO").place(x=100, y=85)


Morfo = tk.Spinbox(master=parametros, from_=0, to=300, textvariable =Morfo , width = 5).place(x=200, y=115)
Morfologia = tk.Button(master=parametros, textvariable = M,text="FM= ", command = obtener, width = 3).place(x=50, y=115)
d = tk.Label(master=parametros, text="MORFOLOGÍA").place(x=100, y=115)

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


#PUNTOS ai bi

#frame en el frame



ab = tk.Frame(master=puntos)
ab.place(x=10,y=20)
ab.config(width=350, height=200, bd=8, highlightbackground="black", highlightthickness=1)



#pequeños frames en el frame
cua = tk.Frame(master=ab)
cua.place(x=30, y=5)
cua.config( width=55, height=50,bd=8, highlightbackground="black",highlightthickness=1)
e= tk.Label(master= cua, text= "ai").place(x=10,y=10)

cua1 = tk.Frame(master=ab)
cua1.place(x=30, y=55)
cua1.config( width=50, height=50,bd=8, highlightbackground="black",highlightthickness=1)
f= tk.Label(master= cua1, text= "bi").place(x=10,y=10)

cua2= tk.Frame(master=ab)
cua2.place(x=80, y=5)
cua2.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua3= tk.Frame(master=ab)
cua3.place(x=80, y=55)
cua3.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua4= tk.Frame(master=ab)
cua4.place(x=130, y=5)
cua4.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua5= tk.Frame(master=ab)
cua5.place(x=130, y=55)
cua5.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua6 = tk.Frame(master=ab)
cua6.place(x=180, y=5)
cua6.config( width=50, height=50,bd=8, highlightbackground="black",highlightthickness=1)

cua7 = tk.Frame(master=ab)
cua7.place(x=180, y=55)
cua7.config( width=50, height=50,bd=8, highlightbackground="black",highlightthickness=1)

cua8= tk.Frame(master=ab)
cua8.place(x=230, y=5)
cua8.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua9= tk.Frame(master=ab)
cua9.place(x=230, y=55)
cua9.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua10= tk.Frame(master=ab)
cua10.place(x=280, y=5)
cua10.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)

cua11= tk.Frame(master=ab)
cua11.place(x=280, y=55)
cua11.config( width=50, height=50,  bd=8, highlightbackground="black",highlightthickness=1)
#METODOS SOLUCION"""

window.mainloop()

##

