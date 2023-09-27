#---------------Importar  Librerias-----------------------
import cv2
import os
from ultralytics import YOLO

#-------------IMPORTAR LA CLASE SEGUIMIENTO MANO-------------------
import SeguimientoManos as sm


#----------Lectura de la camara --------------------------
cap =cv2.VideoCapture(0)
#----Cambiar la Resolucion de la camara------
cap.set(3,1280)
cap.set(4,720)

#---LEER NUESTRO MODELOS --
model = YOLO('Train_David.pt')
#model = YOLO('abecedario.pt')
#MODIFICAR
#modelo1=YOLO()


#--------DECLARAR NUESTRO DETECTOR------------

detector = sm.detectormanos(Confdeteccion=0.9)

while True:
    #----Realiza la lectura----
    ret,frame=cap.read()
    #----Extraer informacion de la mano---------
    frame=detector.encontrarmanos(frame,dibujar=False)  #---EN ESTA PARTE ES LA QUE DETECTA LA MANO Y LA DIBUJA --

    #----POSICION DE UNA SOLA MANO----
    lista1,bbox,mano=detector.encontrarposicion(frame,ManoNum=0,dibujarPuntos=False,dibujarBox=True,color=[0,255,0])

    if mano== 1:
        #----Extraer la informacion del cuadro ----
        xmin,ymin,xmax,ymax=  bbox

        #-----Asignamos un margen al cuadro ----
        xmin=xmin-120
        ymin=ymin-120
        xmax=xmax+120
        ymax=ymax+120

        #----Realizamos un recorte de nuestra mano -----
        recorte=frame[ymin:ymax,xmin:xmax]

        #-----REDIMENSIONAMOS LA CAPTURA DE LA MANO --
        recorte= cv2.resize(recorte,(640,640),interpolation=cv2.INTER_CUBIC)

        #----extraer los resultados ---

           #PARA MODELO YOLOV8N  0.33
           #PARA MODELO YOLOV8S  0.44
           #PARA MODELO YOLOV8M  0.55
           #PARA MODELO YOLOV8L  0.75
           #PARA MODELO YOLOV8X  0.85

        resultados =model.predict(recorte,conf=0.75)
        #if resultados==0:


        #----si hay resultado -----

        if(len(resultados)!=0):
            #--iteramos
            for results in resultados:
                masks=results.masks
                coordenadas=masks

                anotaciones=resultados[0].plot()

        cv2.imshow("RECORTE", anotaciones)
        #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)

    #-----------MOSTRAR LOS FPS------------------
    cv2.imshow("LENGUAJE ABECEDARIO",frame)

    #---LEER NUESTRO TECLADO ------s
    t=cv2.waitKey(1)
    if t==27:
        break
cap.release()
cv2.destroyAllWindows()


