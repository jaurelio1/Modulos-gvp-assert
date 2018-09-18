import numpy as np
import cv2
import threading
import time

ThresholdBinarizacao = 20
area = 0
w = 0
h = 0
borda = 200
areaMinContorno = 3000
cont_carros = 0
cont_entradas = 0
cont_saidas = 0


def TestaInterseccaoEntrada(x, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida):
    DiferencaAbsoluta = abs(x - CoordenadaXLinhaSaida)

    if ((DiferencaAbsoluta >= 2) and (x < CoordenadaXLinhaEntrada)):
        return 1
    else:
        return 0

def TestaInterseccaoSaida(x, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida):
    DiferencaAbsoluta = abs(x - CoordenadaXLinhaEntrada)

    if ((DiferencaAbsoluta >= 2) and (x < CoordenadaXLinhaSaida)):
        return 1
    else:
        return 0

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

primeiroFrame = None

for i in range(0, 100):
    (grabbed, Frame) = camera.read()

while True:
    (grabbed, Frame) = camera.read()
    height = np.size(Frame, 0)
    width = np.size(Frame, 1)

    hsv = cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)
    #cv2.imshow("HSV", hsv)
    if not grabbed:
        break

    frameGray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Color Gray", frameGray)
    frameGray = cv2.GaussianBlur(frameGray, (21, 21), 0)

    if primeiroFrame is None:
        primeiroFrame = frameGray
        continue

    FrameDelta = cv2.absdiff(primeiroFrame, frameGray)

    #FrameThresh = cv2.threshold(FrameDelta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    FrameThresh = cv2.threshold(FrameDelta, ThresholdBinarizacao, 255, cv2.THRESH_BINARY)[1]

    FrameThresh = cv2.dilate(FrameThresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(FrameThresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("Imagem Dilatada", FrameThresh)

    quantContornos = 0
    quantBigCar = 0
    quantSmallCar = 0

    CoordenadaXLinhaSaida = (width // 2) - borda
    CoordenadaXLinhaEntrada = (width // 2) + borda

    cv2.line(Frame, (CoordenadaXLinhaEntrada, 0), (CoordenadaXLinhaEntrada, height), (255, 0, 0), 2)
    cv2.line(Frame, (CoordenadaXLinhaSaida, 0), ((CoordenadaXLinhaSaida), height), (0, 0, 255), 2)

    for c in cnts:
        if cv2.contourArea(c) < areaMinContorno:
            continue
        quantContornos += 1

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if (cv2.contourArea(c) < 15000):
            quantSmallCar += 1
            cv2.drawContours(Frame, [box], 0, (0, 255, 0), 2)
        elif (15000 < cv2.contourArea(c) < 37000):
            quantBigCar += 1
            cv2.drawContours(Frame, [box], 0, (255, 0, 0), 2)

        (y, x, h, w) = cv2.boundingRect(c)

        CoordenadaXCentroContorno = (2*x + w) // 2
        CoordenadaYCentroContorno = (2*y + h) // 2
        PontoCentralContorno = (CoordenadaYCentroContorno, CoordenadaXCentroContorno)
        cv2.circle(Frame, PontoCentralContorno, 1, (0, 0, 0), 5)

        print(TestaInterseccaoEntrada(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida)
              , TestaInterseccaoSaida(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida))
        if(TestaInterseccaoEntrada(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida)):
            cont_entradas += 1

        if(TestaInterseccaoSaida(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida)):
            cont_saidas += 1


    '''if (cont_entradas == TestaInterseccaoSaida(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida)
            and quantContornos > 0):
        cont_carros += 1
        continue'''

    cv2.putText(Frame, "Quant Contornos: {}".format(str(quantContornos)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(Frame, "Veiculos Pequenos: {}".format(str(quantSmallCar)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 1), 2)
    cv2.putText(Frame, "Veiculos Grandes: {}".format(str(quantBigCar)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)

    cv2.putText(Frame, "Quantidade de Veiculos: {}".format(str(cont_carros)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)

    #print (cont_entradas, cont_saidas)

    cv2.imshow("Original", Frame)

    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()