import numpy as np
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