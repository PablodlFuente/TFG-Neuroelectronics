# -*- coding: utf-8 -*-
'''
AUTHOR: Pablo de la Fuente Fernández

DESCRIPTION: el presente código toma como input una serie de fotografías de varios patrones circulares y
calcula la distancia entre ellos. La finalidad de este código es seguir un mismo procedimiento para el anális de todas las imágenes, de
manera que se eviten errores sistematicos si se hiciese de forma manual.

REQUISITOS: Se mostrará la imagen con las detecciones que ha hecho el programa de los centroides, si el programa los ha detectado mal por alguna carácteristica de la imagen (falta de contraste, excesivo brilllo, etc) esta puede ser desechada pulsando la tecla "DEL" . Para dar como buena imagen es necario pulsar la tecla "ESC"

El resultado se guarda en un archivo csv cuya primera columna son los diámetros y la segunada las desviaciones estandar de estos.
'''
import numpy as np
import cv2
import os
from scipy.spatial import distance
import pandas as pd

for i in files_names[:]:
    imagen = cv2.imread(dir_data + i)
    if (imagen is None):
        print("Error: no se ha podido encontrar la imagen")
        quit()

    # Convertimos la imagen a HSV
    image_data = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # image_data_Noiseto0 = cv2.threshold(image_data, np.max(image_data) / 4, 255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(image_data, (5, 5), 0)
    ret3, image_data_Noiseto0 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Invertimos la mascara para obtener las bolas
    bolas = image_data_Noiseto0

    # Eliminamos ruido
    kernel = np.ones((3, 3), np.uint8)
    bolas = cv2.morphologyEx(bolas, cv2.MORPH_OPEN, kernel)
    bolas = cv2.morphologyEx(bolas, cv2.MORPH_CLOSE, kernel)

    # Buscamos los contornos de las bolas y los dibujamos en verde
    contours, _ = cv2.findContours(bolas, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(imagen, contours, -1, (0, 255, 0), 2)

    # Buscamos el centro de las bolas y lo pintamos en rojo
    centr = []
    for j in contours:
        # Calcular el centro a partir de los momentos
        momentos = cv2.moments(j)
        if momentos['m00'] != 0:
            cx = int(momentos['m10'] / momentos['m00'])
            cy = int(momentos['m01'] / momentos['m00'])
            centr.append((cx, cy))
            # Dibujar el centro
            cv2.circle(imagen, (cx, cy), 3, (0, 0, 255), -1)

            # Escribimos las coordenadas del centro
            cv2.putText(imagen, "(x: " + str(cx) + ", y: " + str(cy) + ")", (cx + 10, cy + 10), font, 0.5,
                        (255, 255, 255), 1)

    # Mostramos la imagen final
    cv2.imshow(i, imagen)

    # Salir con ESC
    while (1):
        tecla = cv2.waitKey(5)
        if tecla == 27:  # esc key para  imagen ok
            centroides.append(centr)
            distancias.append(np.sort(distance.pdist(centroides[-1], 'euclidean')))
            num_centroids = len(centroides[-1])
            promedio = np.average(distancias[-1][:num_centroids])
            std = np.std(distancias[-1][:num_centroids])
            results.append([promedio, std])
            print(promedio, std/np.sqrt(num_centroids))
            break
        elif tecla == 0:  # del key para imagen no ok
            break
    cv2.destroyAllWindows()

# Cargamos una fuente de texto
font = cv2.FONT_HERSHEY_SIMPLEX
dir_data = #"[dirección donde se localizan las imágenes]"
files_names = os.listdir(dir_data)
centroides = []
distancias = []
results = []
np.savetxt('results.csv', results, delimiter=',')
cv2.destroyAllWindows()
