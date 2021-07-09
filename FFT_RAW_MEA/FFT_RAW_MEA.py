import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

path = "C:/Users/Pabloo/Desktop/NEURONAS TFG/data_txt/"

for i in os.listdir(path):
    NAME = "data_txt/"+i
    electrodos = ['a', 'b', '3', '4', 'c', 'd', 'e', '8', 'f', 'g', '11', 'h', '13',
                  '14', 'i', 'j', '17', '18', 'k', '20', 'l', '22', '23', '24', '25',
                  '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36',
                  '37', '38', '39', 'm', '41', 'n', '43', '44', 'ñ', 'o', '47', '48',
                  'p', '50', 'q', '52', '53', 'r', 's', 'u', '57', '58', 'v', '60']
    file = pd.read_csv(NAME, skiprows=4, delimiter='\t', names=['t'] + electrodos)
    tf = file['t'].iloc[-1]
    file['t'] /= tf
    file.iloc[::1000].plot(x='t', subplots=True, layout=(6, 10), title=NAME, ylabel="V($\mu V$)", sharey=True)

    n = len(file)

    Ts = (file['t'][1] - file['t'][0]) * tf / 1000
    volt = file['s']

    freq = np.fft.rfftfreq(n, d=Ts)
    freq_shift = np.fft.fftshift(freq)
    '''
    Otra manera de generar freq_shift es :
    Fs = 1 / Ts
    freq_shift = np.arange(-Fs / 2, Fs / 2, Fs / n)
    '''

    FFT = np.fft.rfft(volt)  # no se por qué lo multiplico por Ts, de hecho yo creo que está mal
    FFT_shift = np.fft.fftshift(FFT) / n  # normalización y shift de FFT
    plt.figure(NAME)
    plt.subplot(121)
    plt.plot(freq, np.abs(FFT))
    plt.xlim(0, 1000)
    plt.xlabel("f (Hz)")
    plt.ylabel("I(u.a)")
    plt.legend('h')
    plt.ylim(0, 2 * 10 ** 8)
    ####################

    volt = file['53']
    FFT = np.fft.rfft(volt)  # no se por qué lo multiplico por Ts, de hecho yo creo que está mal
    FFT_shift = np.fft.fftshift(FFT) / n  # normalización y shift de FFT

    # fft_high_pass=np.where(np.abs(freq)<=100, 0, fft) # filtro para frecuencias menores de 100
    plt.subplot(122)
    plt.plot(freq, np.abs(FFT))
    plt.xlim(0, 1000)
    plt.xlabel("f (Hz)")
    plt.ylabel("I(u.a)")
    plt.legend(['11'])
    plt.ylim(0, 2 * 10 ** 8)
