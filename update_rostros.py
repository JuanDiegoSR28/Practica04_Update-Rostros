import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

#Si las carpetas de la base de datos no empiezan numeradas en 0, entonces no se coloca un nombre en el índice 0 de la lista.
subjects = ["", "Pepe", "Juan","Jose","Miguel","Toño","Manuel", "Adrian","Jesús","Adolfo"]
#subjects = ["", "1", "2","3","4","5","6", "7","8","9"]

def detect_face(img):
    #Se convierte la imagen a escala de grises para ser usada por el detector
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    #Se lee el modelo de clasificador. 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 
    #Se detectan rostros y se almacenan en la variable faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
 
    #Si no se detecta rostros se regresa la imagen original
    if (len(faces) == 0):
        return None, None
 
    #Suponemos que solo tendremos un rostro
    #Se extra el área del rostro
    (x, y, w, h) = faces[0]
 
    #Se regresa únicamente la parte del rostro en la imagen
    return gray[y:y+w, x:x+h], faces[0]


def prepare_training_data(data_folder_path):
    
    #------Paso 1--------
    #Obten las carpetas dentro de la ruta con las imagenes de entrenamiento. Cada carpeta corresponde a un sujeto.
    dirs = os.listdir(data_folder_path)
    
    #Lista que contendrá las caras de todos los sujetos
    faces = []
    #Lista que contendrá las etiquetas de todos los sujetos
    labels = []
    
    # Navegamos a traves de todas las carpetas y leemos las imágenes dentro de cada carpeta
    for dir_name in dirs:
        
        # Las carpetas de cada sujeto empiezan con la letra 's', por lo tanto, cualquier carpeta que no empeza con s se ignora
        if not dir_name.startswith("s"):
            continue;
            
        #------Paso 2--------
        # Se extrae el número de etiqueta del sujeto a partir del nombre de la carpeta
        # Recuerda el formato de la carpeta = slabel
        # Removemos la letra "s" y nos quedamos solo con el número (Esto es la etiqueta)        
        label = int(dir_name.replace("s", ""))
        
        # Construimos la ruta hacia la carpeta que contiene las imagenes del sujeto actual
        # Ejemplo de ruta = "./images/traindata/s1"
        subject_dir_path = data_folder_path + "/" + dir_name
        
        #Se obtienen los nombres de las imagenes que estan dentro de cada carpeta
        subject_images_names = os.listdir(subject_dir_path)
                #------Paso 3--------
        #Navegamos a través de cada imagen y la leemos
        #detectamos los rostros y los agregamos a la lista
        for image_name in subject_images_names:
            
            #Para ignorar archivos del sistema como .DS_Store
            if image_name.startswith("."):
                continue;
            
            #se construye la ruta de la imagen que se va a leer
            #Ejemplo de ruta = ./images/traindata/s1/1.jpg
            image_path = subject_dir_path + "/" + image_name

            #leer la imagen
            image = cv2.imread(image_path)
                      
            #detección de rostro (usando la función definida arriba)
            face, rect = detect_face(image)
            
            #------Paso 4--------
            #Para propositos de este tutorial
            #se ignoran los rostros que no son detectados
            if face is not None:
                #se agrega el rostro a la lista de rostros
                faces.append(face)
                #se agrega la etiqueta a la lista de etiquetas
                labels.append(label)
                
    return faces, labels

print("Preparando datos...")
faces, labels = prepare_training_data("./traindata")
print("Datos preparados")

#Se imprime el total de rostros y etiquetas encontrados
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
