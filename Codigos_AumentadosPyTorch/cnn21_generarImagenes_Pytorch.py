#!/usr/bin/env python3
import math
import os
import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import random


#Clase que permite añadir ruido gaussiano a un patch (en todas las bandas por igual)
class AnhadirRuidoGaussiano(torch.nn.Module):
  #Recibe la media del ruudo y la intensidad
  def __init__(self, mean=0., std=0.02):
    super().__init__()
    self.std = std
    self.mean = mean

  def forward(self, tensor):
    #Generamos un nuevo tensor lleno de valores aleatorios siguiendo una distribución gaussiana con la desviación y media que indicamos y se suma al patch original (tensor)
    noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    #Mantenemos los valores entre 0 y 1
    return torch.clamp(tensor + noise, 0.0, 1.0)

#Clase que permite añadir ruido gaussiano a cada banda de manera independiente
class AnhadirRuidoEspectral(torch.nn.Module):
    #Recibe el rango de ruido con el que puede trabajar en cada una de las bandas del patch
    def __init__(self, std_range=(0.01, 0.03)):
      super().__init__()
      self.std_range = std_range

    def forward(self, tensor):
      #Extraemos las dimensiones del patch
      B, H, V = tensor.size()

      #Creamos una desviación estándar aleatoria (dentro del rango establecido) para cada banda del patch
      stds = torch.empty(B, 1, 1).uniform_(self.std_range[0], self.std_range[1]).to(tensor.device)
      
      #Creamos un patch de ruido del mismo tamaño que el patch y le aplicamos la desviación a cada banda
      noise = torch.randn(tensor.size(), device=tensor.device) * stds

      #Se aplica el ruido al patch
      return torch.clamp(tensor + noise, 0.0, 1.0)

#Clase que permite cambiar la iluminación del patch en todas las bandas    
class IluminacionAleatoria(torch.nn.Module):
    #Recibe el rango en el que puede operar para aumentar o disminuír la iluminación del patch en todas las bandas
    def __init__(self, factor_range=(0.8, 1.2)):
      super().__init__()
      self.factor_range = factor_range

    def forward(self, tensor):
      #Generamos un número aleatorio dentro del rango establecido
      factor = random.uniform(self.factor_range[0], self.factor_range[1])
      #Aplicamos el factor a todo el patch y evitamos que se salga de los valores se salgan de los límites tras normalizar
      return torch.clamp(tensor * factor, 0.0, 1.0)

#Clase que elimina bandas completas del patch (las pone a 0)
class EliminarBandas(torch.nn.Module):
    #Recibe la probabilidad de borrado
    def __init__(self, drop_prob=0.15):
      super().__init__()
      self.drop_prob = drop_prob

    def forward(self, tensor):
      #Obtenemos las dimensiones del patch
      B, H, V = tensor.size()
      #Creamos una máscara aleatoria de 0s y 1s para las bandas
      mask = (torch.rand(B, 1, 1) > self.drop_prob).float().to(tensor.device)
      #Multiplicamos el patch por la máscara, haciendo que las bandas que tienen un 0 en la máscara pasen a valer 0
      return tensor * mask
    

def read_raw(fichero):
  #Leemos los 3 primeros números presentes en el archivo, los 3 son enteros de 32 bits
  #B es el número de bandas
  #H es la anchura en píxeles del dataset
  #V es la altura en píxeles del dataset
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)

  #Se leen todos los datos contenidos en el dataset (un total de B*H*V enteros de 32 bits)
  #Se saltan los primeros 12 bytes correspondientes a la cabecera B,H,V
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32).astype(np.float32)
  
  # normalizar
  #Se realiza el normalizado de los datos empleando la escala Min-Max para transformar todos los valores al rango [0,1]
  d_min = datos.min()
  d_max = datos.max()
  datos -= d_min
  datos /= (d_max - d_min)

  #Se reestructura el array de datos leídos del fichero en un bloque con 3 dimensiones, el alto (V), el ancho (H) y la banda (B)
  datos=datos.reshape(V,H,B)

  #Se convierte el objeto de Numpy en un Tensor de PyTorch de fipo Float (de 32 bits)
  #Esto se hace para poder emplear los datos en la red neuronal en PyTorch, además permiten ser movidos a la GPU para poder ser procesados
  datos=torch.FloatTensor(datos)

  #Devolvemos los datos junto a los valores de ancho, alto y número de bandas
  return(datos,H,V,B)

#Función que obtiene un patch a partir de un píxel concreto (un centroide de un segmento), crea el patch alrededor del píxel dado.
#Sizex y sizey establecen el tamaño del patch
#x e y son las coordenadas del píxel central
def select_patch(datos,sizex,sizey,x,y):
  #Calculamos las coordenadas de las esquinas del cuadro
  
  #X1 e Y1 son la parte superior izquierda del patch
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));
  
  #X2 e Y2 son la parte inferior derecha del patch
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  
  #Se realiza el slice sobre los datos, obteniendo el patch
  patch=datos[:,y1:y2,x1:x2]
  return(patch)


#Función que permite guardar el mapa de clasificación final en un nuevo fichero
def save_raw(output,H,V,B,filename):
  #Tratamos de abrir el fichero en modo escritura binaria
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    #Se crea un array con los datos sobre el número de bandas (B), el número de píxeles de anchura (H) y el número de píxeles de altura (V), los cuales son enteros de 32 bits
    sizes=np.array([B,H,V], dtype=np.uint32)
    #Se transforman los datos desde el cubo de 3 dimensiones a una única fila y se desconectan lso datos d ela red neuronal de PyTorch con detach()
    output=output.reshape(H*V*B).flatten().detach()
    #Transformamos el tensor de PyTorch en un array de NumPy de enteros de 32 bits
    output=output.numpy().astype(np.uint32)
    #Pegamos la cabecera delante de los datos de la imagen
    output=np.concatenate([sizes,output])
    #Se vuelcan los datos en el fichero
    output.tofile(f,format="%d")
    f.close()
    print('* Archivo raw guardado',filename)


#Función que permite almacenar un patch concreto que estea siendo procesado
def save_patch(datos,sizex,sizey,B,filename):
  #Se mueven los datos a la RAM principal desde la VRAM
  datos=datos.cpu()

  #Para guardar los patches cuyos valores han sido normalizados a valores decimales debemos multiplicar a 255 para que no sean todos 0 y 1 al usar la función save_raw
  if datos.max() <= 1.0:
    datos = datos * 255

  #Se reorganizan los datos para poder ser almacenados en el formato usado en los ficheros de datos (alto, ancho, bandas) en lugar de (bandas, alto, ancho) usado por pytorch
  datos=np.transpose(datos,(1,2,0))
  #Se almacena el patch usando la función save_raw
  save_raw(datos,sizex,sizey,B,filename)

# --- MAIN ---
if __name__ == '__main__':
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    DATASET = '/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.raw'
    
    #Leemos el dataset
    (datos_raw, H, V, B)= read_raw(DATASET)
    datos_tensor = datos_raw.permute(2, 0, 1).contiguous()
    
    sizex, sizey = 32, 32
    
    #Obtenemos el patch sobre el que realizaremos las transformaciones (COORDENADAS ACTUALIZADAS)
    patch_original = select_patch(datos_tensor, sizex, sizey, 4500, 3000)

    #Creamos la lista de aumentado base con flips horizontales y verticales (en este caso con probabilidad 1 para que se apliquen sí o sí a las imágenes)
    flips_list = [v2.RandomHorizontalFlip(p=1.0), v2.RandomVerticalFlip(p=1.0)]

    #Rotación (Fijada a 45 grados para que sea visible de manera clara)
    #Calculamos un padding suficiente para que al rotar 32x32 no queden huecos, el padding serán el reflejo de los píxeles de los bordes para no generar firmas espectrales totalmente artificiales.
    #La diagonal de 32x32 es aprox 45. Un padding de 8 a cada lado nos da 48x48, necesario para que al hacer rotaciones no queden huecos en los datos
    #Tras hacer la rotación se hace un recorte a partir del centro para devolver el parche al tamaño original tras haberlo rotado
    rotation = v2.Compose([
        v2.Pad(padding=8, padding_mode='reflect'),
        v2.RandomRotation(degrees=(45, 45), interpolation=InterpolationMode.NEAREST),
        v2.CenterCrop(size=(sizey, sizex))
    ])

    #Zoom (Fijado a 0.8, un zoom-out que se puede ver de manera clara)
    simetric_zoom = v2.RandomAffine(degrees=0, scale=(0.8, 0.8))
    
    #Ruido gaussiano
    noise = AnhadirRuidoGaussiano(std=0.02)
    #Ruido espectral
    spec_noise = AnhadirRuidoEspectral(std_range=(0.03, 0.03)) #Forzado al máximo del rango para el ejemplo
    #Iluminación aleatoria
    spec_illum = IluminacionAleatoria(factor_range=(0.8, 0.8)) #Forzado a oscurecer para el ejemplo
    #Eliminación de bandas aleatoria
    spec_drop = EliminarBandas(drop_prob=0.15)
    
    #Borrado Aleatorio (p=1.0 garantizado)
    erasing = v2.RandomErasing(p=1.0, scale=(0.05, 0.05), value=0)
    
    #Generamos un diccionario con 
    transformaciones = {
        "00_Original": None,
        "00_Flips": v2.Compose(flips_list),
        "01_Rotacion": v2.Compose([rotation]),
        "02_Zoom": v2.Compose([simetric_zoom]),
        #"03_Rotacion_Zoom": v2.Compose([rotation, simetric_zoom]),
        "04_Ruido_Gaussiano": v2.Compose([noise]),
        "05_Ruido_Espectral": v2.Compose([spec_noise]),
        #"06_R_Gaussiano_R_Espectral": v2.Compose([noise, spec_noise]),
        "07_Iluminacion_Aleatoria": v2.Compose([spec_illum]),
        "08_Eliminar_Bandas": v2.Compose([spec_drop]),
        #"09_Ilum_Aleatoria_Elim_Bandas": v2.Compose([spec_illum, spec_drop]),
        "10_Borrado_Aleatorio": v2.Compose([erasing]),
        #"11_Rotacion_Borrado": v2.Compose([rotation, erasing]),
        #"12_Rotacion_Zoom_Borrado": v2.Compose([rotation, simetric_zoom, erasing]),
        #"13_Rotacion_Ruido_Espectral": v2.Compose([rotation, spec_noise]),
        #"14_Rotacion_Zoom_R_Espectral": v2.Compose([rotation, simetric_zoom, spec_noise]),
        #"15_R_Espectral_Borrado": v2.Compose([spec_noise, erasing])
    }
    
    carpeta_salida = "ejemplos_aumentados_raw"
    os.makedirs(carpeta_salida, exist_ok=True)
    
    #Aplicamos todos los aumentados al parche original y almacenamos los resultados
    print("Aplicando aumentados y generando archivos...")
    for nombre, transformacion in transformaciones.items():
        if transformacion is None:
            patch_transformado = patch_original.clone()
        else:
            patch_transformado = transformacion(patch_original)
            
        ruta_salida = os.path.join(carpeta_salida, f"{nombre}.raw")

        save_patch(patch_transformado, sizex, sizey, B, ruta_salida)
        
    print(f"\nSe han generado todas las imágenes")