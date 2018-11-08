import numpy as np
import cv2

ThresholdBinarizacao = 20
borda = 200
areaMinContorno = 3000
thickness = 2

cont_carros = 0
cont_saidas = 0

quantBigCar = 0
quantSmallCar = 0
quantMoto = 0

times = 0
times_entrada = 0

def TestaInterseccaoEntrada(x, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida):
    global times_entrada
    DiferencaAbsoluta = abs(x - CoordenadaXLinhaSaida)

    if (x < CoordenadaXLinhaEntrada):
        times_entrada += 1

def TestaInterseccaoSaida(x, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida):
    global cont_saidas, times
    DiferencaAbsoluta = abs(x - CoordenadaXLinhaEntrada)

    if ((DiferencaAbsoluta >= 2) and (x <= CoordenadaXLinhaSaida + 5)):
        cont_saidas += 1

    if(x < CoordenadaXLinhaSaida + 5):
        times += 1

    if cont_saidas == 1 and times <= 1:
        return 1
    else:
        return 0

def TestaVeiculo(h, w, c):
    tipo = ""
    contorno = cv2.contourArea(c)
    if (5000 < contorno < 15000):
        tipo+= "moto"
    elif (15000 < contorno < 37000):
        razao = h/w
        if(0.3<razao<=0.51):
            tipo += "carro"
        elif(razao <= 0.3):
            tipo += "veiculo grande"

    return tipo

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

    kernel = np.ones((5,5), np.uint8)

    #hsv = cv2.cvtColor(Frame, cv2.COLOR_BGR2HSV)   ver se ainda vai necessitar
    if not grabbed:
        break

    frameGray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.GaussianBlur(frameGray, (21, 21), 0)

    if primeiroFrame is None:
        primeiroFrame = frameGray
        continue

    FrameDelta = cv2.absdiff(primeiroFrame, frameGray)

    FrameThresh = cv2.threshold(FrameDelta, ThresholdBinarizacao, 255, cv2.THRESH_BINARY)[1]

    opening = cv2.morphologyEx(FrameThresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    FrameThresh = cv2.dilate(closing, kernel, iterations=2)
    cv2.imshow('FrameThresh', FrameThresh)
    _, cnts, _ = cv2.findContours(FrameThresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    quantContornos = 0

    CoordenadaXLinhaSaida = int(width / 2) - borda
    CoordenadaXLinhaEntrada = int(width / 2) + borda

    cv2.line(Frame, (CoordenadaXLinhaEntrada, 0), (CoordenadaXLinhaEntrada, height), (255, 0, 0), thickness)
    cv2.line(Frame, (CoordenadaXLinhaSaida, 0), ((CoordenadaXLinhaSaida), height), (0, 0, 255), thickness)

    for c in cnts:
        if cv2.contourArea(c) < areaMinContorno:
            continue
        quantContornos += 1

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        (x, y, w, h) = cv2.boundingRect(c)

        CoordenadaXCentroContorno = (2*x + w) // 2
        CoordenadaYCentroContorno = (2*y + h) // 2
        PontoCentralContorno = (CoordenadaXCentroContorno, CoordenadaYCentroContorno)
        cv2.circle(Frame, PontoCentralContorno, 1, (0, 0, 0), 5)

        veiculo = TestaVeiculo(h, w, c)
        TestaInterseccaoEntrada(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida)

        #red
        if veiculo == 'moto':
            cv2.drawContours(Frame, [box], 0, (0, 0, 255), thickness)
            if times_entrada == 1:
                quantMoto += 1
        #green
        elif veiculo == 'carro':
            cv2.drawContours(Frame, [box], 0, (0, 255, 0), thickness)
            if times_entrada == 1:
                quantSmallCar += 1
        #blue
        elif veiculo == 'veiculo grande':
            cv2.drawContours(Frame, [box], 0, (255, 0 ,0), thickness)
            if times_entrada == 1:
                quantBigCar += 1

        if (TestaInterseccaoSaida(CoordenadaXCentroContorno, CoordenadaXLinhaEntrada, CoordenadaXLinhaSaida)):
            cont_carros += 1

    if quantContornos == 0:
        times = 0
        times_entrada = 0
        cont_saidas = 0

    cv2.putText(Frame, "Motos: {}".format(str(quantMoto)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2)
    cv2.putText(Frame, "Veiculos Pequenos: {}".format(str(quantSmallCar)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 1), 2)
    cv2.putText(Frame, "Veiculos Grandes: {}".format(str(quantBigCar)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 0, 0), 2)

    cv2.putText(Frame, "Total de Veiculos: {}".format(str(cont_carros)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 2)

    cv2.imshow("Original", Frame)

    cv2.waitKey(1)

camera.release()
cv2.destroyAllWindows()