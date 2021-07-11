# -*- coding: utf-8 -*-
'''
AUTHOR: Pablo de la Fuente Fernández

DESCRIPTION: El presente código toma como entrada una serie de gráficas obtenidas por un perfilómetro, estas son 
analizadas obteneindo la altuma máxima que hay previamnete suprimiendo las posibles derevaciones que existen en los 
datos.
 Las graficas deben pertenecer a alguno de los siguiente tipos:
 
1 Escalón: se procesa una gráfica escalón tanto de subida como de bajada.
2 Trinchera: se procesa una gráfica escalón tanto de subida como de bajada. 
3 Columnas: se procesa una gráfica que tenga picos (columnas) 

REQUISITOS: se pedirá al usuario que seleccione sobre la imagen dónde están situados el/los salto/s. En el caso de las columnas se pedirá donde comienazan y terminan dichas columnas.

El programa devuelve un archivo .csv con los resultados de la altura del escalón y las nuevas imágenes corregidas
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.widgets import Cursor
import ctypes

def Mbox(title, text, style=0):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def create_plot(data, name):
    global fig
    global ax

    fig = plt.figure(name)
    ax = fig.subplots()
    ax.plot(data[:, 0], data[:, 1])


def onclick(event):
    global coord
    coord.append((event.xdata, event.ydata))
    x = event.xdata
    y = event.ydata

    # printing the values of the selected point
    print([x, y])
    annot.xy = (x, y)
    text = "({:.2g}, {:.2g})".format(x, y)
    annot.set_text(text)
    annot.set_visible(True)
    fig.canvas.draw()  # redraw the figure


def leer_archivo(loc, descartar_dcha=-1, descartar_izda=0):
    file = pd.read_csv(loc, delimiter='\t', skiprows=40, index_col=False, header=None, skipfooter=2)
    file = file.drop([5], axis=1) / 10

    file = file.to_numpy()
    file = file.reshape((np.product(file.shape), 1))[
           descartar_izda:descartar_dcha]  # esto de [:1460] solo es para los nuestros
    x = np.linspace(0, len(file) + 1, len(file)) / 10
    x.shape = (len(x), 1)
    data = np.concatenate((x, file), axis=1)
    return data


def corregir_trinchera(data):
    global annot
    global cursor
    global coord

    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)
    # Creating an annotating box
    annot = ax.annotate("", xy=(0, 0), xytext=(-40, 40), textcoords="offset points",
                        bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                        arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False)
    # Function for storing and showing the clicked values
    coord = []

    fig.canvas.mpl_connect('button_press_event', onclick)
    Mbox('Información', 'Selecciona el inicio y el final de la trinchera')
    while len(coord) < 2:
        plt.waitforbuttonpress(timeout=- 1)

    selec_signal = np.asarray(coord[:2])

    selec_signal = (selec_signal * 10).astype('int32')

    x_izda_signal = selec_signal[0, 0]
    x_dcha_signal = selec_signal[1, 0]

    popt_izda, cov_izda = curve_fit(func, data[:x_izda_signal, 0], data[:x_izda_signal, 1])
    data[:x_izda_signal, 1] -= (func(data[:x_izda_signal, 0], *popt_izda))
    data[x_izda_signal:, 1] -= (func(data[:x_izda_signal, 0], *popt_izda)[-1])

    popt_dcha, cov_dcha = curve_fit(func, data[x_dcha_signal:, 0], data[x_dcha_signal:, 1])
    data[x_dcha_signal:, 1] -= (func(data[x_dcha_signal:, 0], *popt_dcha))
    data[x_izda_signal:x_dcha_signal, 1] -= (func(data[x_dcha_signal:, 0], *popt_dcha)[0])

    popt_sig, cov_sig = curve_fit(func, data[x_izda_signal:x_dcha_signal, 0], data[x_izda_signal:x_dcha_signal, 1])
    data[x_izda_signal:x_dcha_signal, 1] -= (func(data[x_izda_signal:x_dcha_signal, 0], *popt_sig) - np.average(
        data[x_izda_signal:x_dcha_signal, 1]))

    minimos = data[x_izda_signal:x_dcha_signal, 1]

    return data, minimos


def corregir_escalon(data):
    global annot
    global cursor
    global coord

    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)
    # Creating an annotating box
    annot = ax.annotate("", xy=(0, 0), xytext=(-40, 40), textcoords="offset points",
                        bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                        arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False)
    # Function for storing and showing the clicked values
    coord = []

    fig.canvas.mpl_connect('button_press_event', onclick)
    Mbox('Información', 'Selecciona el inicio del escalón')
    plt.waitforbuttonpress(timeout=- 1)

    selec_signal = np.asarray(coord[:2])

    selec_signal = (selec_signal * 10).astype('int32')

    jump = selec_signal[0, 0]

    popt_izda, cov_izda = curve_fit(func, data[0:jump, 0], data[0:jump, 1])
    data[jump:, 1] -= func(data[:jump, 0], *popt_izda)[-1]
    popt_dcha, cov_dcha = curve_fit(func, data[jump:, 0], data[jump:, 1])
    data[:jump, 1] -= (func(data[:jump, 0], *popt_izda) - func(data[:jump, 0], *popt_izda)[0])
    data[jump:, 1] -= (func(data[jump:, 0], *popt_dcha) - func(data[jump:, 0], *popt_dcha)[0])

    data[:, 1] -= np.average(data[:jump, 1])

    minimos = data[jump:, 1]

    return data, minimos


def analisis_columnas(data, min_altura_columna=None, min_ancho_columna=None, max_altura_columna=None,
                      max_ancho_columna=None):
    global annot
    global cursor
    global coord

    cursor = Cursor(ax, horizOn=True, vertOn=True, useblit=True,
                    color='r', linewidth=1)
    # Creating an annotating box
    annot = ax.annotate("", xy=(0, 0), xytext=(-40, 40), textcoords="offset points",
                        bbox=dict(boxstyle='round4', fc='linen', ec='k', lw=1),
                        arrowprops=dict(arrowstyle='-|>'))
    annot.set_visible(False)
    # Function for storing and showing the clicked values
    coord = []

    fig.canvas.mpl_connect('button_press_event', onclick)
    Mbox('Información', 'Selecciona el inicio y el final de la aparición de las columnas. Se permiten valores fuera de los limites de la gráfica.')
    while len(coord) < 2:
        plt.waitforbuttonpress(timeout=- 1)

    selec_signal = np.asarray(coord[:2])

    selec_signal = (selec_signal * 10).astype('int32')

    x_izda_signal = selec_signal[0, 0]
    x_dcha_signal = selec_signal[1, 0]

    peaks, propiedades = find_peaks(data[:, 1], prominence=(min_altura_columna, max_altura_columna),
                                    width=(min_ancho_columna, max_ancho_columna))
    minimuns = find_peaks(-data[:, 1], prominence=(min_altura_columna, max_altura_columna))[0]

    heights = propiedades['prominences']

    popt, cov = curve_fit(func, data[minimuns][:, 0], data[minimuns][:, 1])
    data[:, 1] -= func(data[:, 0], *popt)

    num_datos = len(data[:, 0])

    if x_izda_signal <= num_datos:
        popt_izda, cov_izda = curve_fit(func, data[:x_izda_signal, 0], data[:x_izda_signal, 1])
        data[:x_izda_signal, 1] -= func(data[:x_izda_signal, 0], *popt_izda)

    if x_dcha_signal <= num_datos:
        popt_dcha, cov_dcha = curve_fit(func, data[x_dcha_signal:, 0], data[x_dcha_signal:, 1])
        data[x_dcha_signal:, 1] -= func(data[x_dcha_signal:, 0], *popt_dcha)

    return data, heights


for i in names_files[:]:
    result = leer_archivo(path + i)  # , descartar_dcha=1460)
    create_plot(result, name=i)

    result_corr, mins = corregir_escalon(np.copy(result))
    #result_corr, mins = corregir_trinchera(np.copy(result))
    #result_corr, mins = analisis_columnas(np.copy(result), min_altura_columna=80)

    profundidad.append(np.average(mins))
    std_profundidad.append(np.std(mins))

    plt.cla()
    plt.xlabel("distancia (nm)")
    plt.ylabel("z (nm)")
    ax.plot(result_corr[:, 0], result_corr[:, 1])
    # ax.hlines(profundidad[-1], result[:, 0][0], result[:, 0][-1], 'g')
    # ax.plot(result[:, 0], np.gradient(result[:, 1]))  # derivada
    ax.grid()
    plt.savefig(path + i + '.png')
func = lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d


path = #"[dirección donde se encuentran los datos RAW del perfilómetro]"


names_files = os.listdir(path)

profundidad = []
std_profundidad = []

profundidad = np.asarray(profundidad)
profundidad.shape = (len(profundidad), 1)

std_profundidad = np.asarray(std_profundidad)
std_profundidad.shape = (len(std_profundidad), 1)

resultados = np.concatenate((np.abs(profundidad), std_profundidad), axis=1)
np.savetxt(path, resultados,
           delimiter=',')
