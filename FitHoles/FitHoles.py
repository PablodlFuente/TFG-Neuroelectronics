'''
AUTHOR: Pablo de la Fuente Fernández

DESCRIPTION: el presente código toma como input una serie de fotografías (en formato .fit) de patrones circulares y
calcula el diametro de dichos patrones. La finalidad de este código es seguir un mismo procedimiento para el anális de todas las imágenes, de
manera que se eviten errores sistematicos si se hiciese de forma manual.

El resultado se guarda en un archivo csv cuya primera columna son los diámetros y la segunada las desviaciones estandar de estos.
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import cv2
import os
from scipy import ndimage

def ImageDiameterPatter(dir_data, isPrintFigures=False):

    diametros = []
    std = []
    files_names = os.listdir(dir_data)

    for i in files_names[:]:
        hdu_list = fits.open(dir_data + i)  # abro los fits
        image_data = hdu_list[0].data  # guardo su info
        hdu_list.close()  # cierro la imagan

        image_data_Noiseto0 = np.where(image_data <= np.max(image_data) / 2, 0,
                                       image_data)  # pongo a valor 0 todo lo que sea menor que la mitad de la altura máxima (el ruido lo anulo)
        image_data_Signalto0 = np.where(image_data >= np.max(image_data) / 2, 0,
                                        image_data)  # hago lo inverso que la anterior. La señal la pongo a 0

        noise = image_data_Signalto0.mean()  # primera estimación del ruido será la media de la imagen con la señal a 0

        m = cv2.moments(image_data_Noiseto0)  # calculo el centroide de la distribución
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        print('x=', x, 'y=', y)

        Max_avg = np.average(
            image_data[y - 2:y + 3, x - 2:x + 3])  # calculo el máximo, pero esta vez promediado con sus vecinos
        HalfMaximum = (Max_avg - noise) / 2  # mitad de la potencia de la imagen real

        image_data_Noiseto0 = np.where(image_data <= HalfMaximum, 0,
                                       1)  # ahora si podemos calcular de manera más precisa la imagen solo con la señal mayor que la potencia/2

        k_vert = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # filtro detección contornos verticales
        k_horiz = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # filtro detección contornos horizontales
        edge_vert = ndimage.convolve(image_data_Noiseto0, k_vert, cval=0.0)  # convolución de la imagen con filtro vert
        edge_horiz = ndimage.convolve(image_data_Noiseto0, k_horiz,
                                      cval=0.0)  # convolución de la imagen con filtro horiz

        px_contors = np.unique(
            np.concatenate((np.array(list(zip(*np.nonzero(edge_vert)))), np.array(list(zip(*np.nonzero(edge_horiz)))))),
            axis=0)  # todos los píxeles que hemos detectado de contorno
        distance2centroid = np.linalg.norm(px_contors - [y, x], axis=1) * 2
        diametros.append(np.average(distance2centroid))  # hacemos un promedio de la distancia del centroide a los pixeles de contorno
        std.append(np.std(distance2centroid))

        delta = np.min([int(image_data.shape[1] / 4), image_data.shape[1] - x,
                        x])  # esto es un factor de recorte para las imágenes
        '''SuperGaussian = lambda l, w, I,s, n: I * np.exp(-2 * ((l-s) / w) ** n)+x
        popt, pcov = curve_fit(SuperGaussian, np.linspace(0, 1, len(image_data[y, x - delta:x + delta])),image_data[y, x - delta:x + delta]-noise, bounds=((-np.inf,-np.inf ,-np.inf, 2),(np.inf,np.inf,np.inf, 10)))
        print(popt)
        l=np.linspace(0,1,1000)
        plt.plot(l, SuperGaussian(l,*popt))
        plt.plot(np.linspace(0, 1, len(image_data[y, x - delta:x + delta])), image_data[y, x - delta:x + delta] - noise,
                 '.r')'''
        if isPrintFigures == True:
            delta = np.min([int(image_data.shape[1] / 4), image_data.shape[1] - x, x])#esto es un factor de recorte para las imágenes
            plt.figure()
            plt.imshow(image_data_Noiseto0, cmap='gray')
            plt.imshow(image_data, cmap='gray', alpha=0.8)
            plt.plot(x, y, '.r')

            plt.figure()
            plt.plot(np.linspace(0, 1, len(image_data[y, x - delta:x + delta])), image_data[y, x - delta:x + delta]-noise, '.r')


            plt.figure()
            image_data_Signalto0.shape = (np.prod(image_data_Signalto0.shape),)
            plt.hist(image_data_Signalto0, bins=np.max(image_data_Signalto0))  # find the most probable value , this is our consideration of noise. Remove noise of the image and rest of values take 1 of intensity, this should produce a square function, the wide is the diameter of the hole
            plt.axvline(noise, color='k', linestyle='dashed', linewidth=1)

    diametros = np.sort(np.asarray(diametros))[::-1]
    return diametros, np.array(std)

dir_data = #"[dirección donde se localizan las imágenes]"

diametros_plus_std = ImageDiameterPatter(dir_data)
np.savetxt('results.csv', diametros_plus_std, delimiter=',')
