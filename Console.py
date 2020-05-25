"""Interacción con usuario, pide parámetros para la generación de la señal del ECG, llama las funciones de WaveGenerator
y retorna el resultado."""
import WaveGenerator as WG


import tkinter as tk
window= tk.Tk()

pf= tk.Label(
	text= "Bienvenido a nuestro proyecto EGG"
	foreground="white",  # Set the text color to white
    background="black"  # Set the background color to blac
)
pf.pack()

def salir():
    window.destroy()

button = tk.Button(
    master= window,
    text="salir",
    width=5,
    height=5,
    bg="white",
    fg="red",
    command= salir,
)
window.mainloop()
