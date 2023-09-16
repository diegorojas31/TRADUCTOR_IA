#---------ESTA ES UNA MUESTRA PARA GENERAR VARIAS CAPTURAS DE UNA SEÑA------------------------------
#---------------Importar  Librerias-----------------------
import  cv2
import os


#-------------IMPORTAR LA CLASE SEGUIMIENTO MANO-------------------
import SeguimientoManos as sm

#----------Lectura de la camara --------------------------
cap =cv2.VideoCapture(0)
#----Cambiar la Resolucion de la camara------
cap.set(3,1280)
cap.set(4,720)
#--------DECLARAR NUESTRO DETECTOR------------

detector = sm.detectormanos(Confdeteccion=0.9)

#---DECLARAMOS UN CONTADOR --
cont =0


while True:
    #----Realiza la lectura----
    ret,frame=cap.read()
    #----Extraer informacion de la mano---------
    frame=detector.encontrarmanos(frame,dibujar=False)  #---EN ESTA PARTE ES LA QUE DETECTA LA MANO Y LA DIBUJA --

    #----POSICION DE UNA SOLA MANO----
    lista1,bbox,mano=detector.encontrarposicion(frame,ManoNum=0,dibujarPuntos=False,dibujarBox=False,color=[0,255,0])

    if mano== 1:
        #----Extraer la informacion del cuadro ----
        xmin,ymin,xmax,ymax=  bbox

        #-----Asignamos un margen al cuadro ----
        xmin=xmin-60
        ymin=ymin-60
        xmax=xmax+60
        ymax=ymax+60

        #----Realizamos un recorte de nuestra mano -----
        recorte=frame[ymin:ymax,xmin:xmax]

        #-----REDIMENSIONAMOS LA CAPTURA DE LA MANO --
        #recorte= cv2.resize(recorte,(640,640),interpolation=cv2.INTER_CUBIC)

        if recorte.shape[0]>0 and recorte.shape[1]>0:
            # ----------------ALMACENAMIENTO DE IMAGENES--------------------------------------
            cv2.imwrite("C:/Users/DIEGO ALEXANDER/PycharmProjects/TRADUCTOR_IA/ABECEDARIO/Letra_A/LETRA_A_{}.jpg".format(cont),recorte) #MODIFICAR
            cont = cont + 1
            cv2.imshow("RECORTE", recorte)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), [0, 255, 0], 2)

    #-----------MOSTRAR LOS FPS------------------
    cv2.imshow("LENGUAJE ABECEDARIO",frame)

    #---LEER NUESTRO TECLADO ------s
    t=cv2.waitKey(1)
    if t==27 or cont==300:    #LA VERIABLE CONT PARA LA CANTIDAD DE CAPTURAS QUE TOMARA PARA CERRARSE EL PROGRMA
        break
cap.release()
cv2.destroyAllWindows()