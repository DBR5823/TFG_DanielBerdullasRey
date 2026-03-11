#!/usr/bin/env python3
import math
import os
import numpy as np
import torch
import torchvision.transforms.v2 as v2
import random


#Clase que permite añadir ruído gaussiano a un patch (en todas las bandas por igual)
class AnhadirRuidoGaussiano(torch.nn.Module):
  #Recibe la media del ruuído y la intensidad
  def __init__(self, mean=0., std=0.05):
    super().__init__()
    self.std = std
    self.mean = mean

  def forward(self, tensor):
    #Generamos un nuevo tensor lleno de valores aleatorios siguiendo una distribución gaussiana con la desviación y media que indicamos y se suma al patch original (tensor)

    noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean

    #Mantenemos los valores entre 0 y 1
    return torch.clamp(tensor + noise, 0.0, 1.0)

#Clase que permite añadir ruído gaussiano a cada banda de manera independiente
class AnhadirRuidoEspectral(torch.nn.Module):
    #Recibe el rango de ruído con el que puede trabajar en cada una de las bandas del patch
    def __init__(self, std_range=(0.01, 0.05)):
      super().__init__()
      self.std_range = std_range

    def forward(self, tensor):
      #Extraemos las dimensiones del patch
      B, H, V = tensor.size()

      #Creamos una desviación estándar aleatoria (dentro del rango establecido) para cada banda del patch
      stds = torch.empty(B, 1, 1).uniform_(self.std_range[0], self.std_range[1]).to(tensor.device)
      
      #Creamos un patch de ruido del mismo tamaño que el patch y le aplicamos
      noise = torch.randn(tensor.size(), device=tensor.device) * stds

      #Se aplica el ruido al patch
      return torch.clamp(tensor + noise, 0.0, 1.0)

#Clase que cambia la iluminación del patch en todas las bandas    
class IluminacionAleatoria(torch.nn.Module):
    #Recibe el rango en el que puede operar de luz
    def __init__(self, factor_range=(0.9, 1.1)):
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
    def __init__(self, drop_prob=0.1):
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
  #Se imprime información sobre el dataset leído
  #print('Lectura del dataset*********')
  #print('* Leyendo dataset:',fichero)
  #print('  B (bandas):',B,'H (anchura):',H,'V (altura):',V)
  #print('  Píxeles leídos:',len(datos))
  # esta red no necesita realmente normalizar
  #Se realiza el normalizado de los datos empleando la escala Min-Max para transformar todos los valores al rango [0,1]
  d_min = datos.min()
  d_max = datos.max()
  datos -= d_min
  datos /= (d_max - d_min)
  #print('  Normalización: Valor min:',datos.min(),'Valor max:',datos.max())

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
    
    # Recibimos el d_max (que será 65535.0)
    (datos_raw, H, V, B)= read_raw(DATASET)
    datos_tensor = datos_raw.permute(2, 0, 1).contiguous()
    
    sizex, sizey = 32, 32
    

    patch_original = select_patch(datos_tensor, sizex, sizey, 4100, 3200)

    flips = [v2.RandomHorizontalFlip(p=1.0), v2.RandomVerticalFlip(p=1.0)]
    rotation = v2.RandomRotation(degrees=(45, 45)) # Fijado a 45 grados exactos
    simetric_zoom = v2.RandomAffine(degrees=0, scale=(0.7, 0.7)) # Fijado a un zoom-out obvio
    
    noise = AnhadirRuidoGaussiano(std=0.1) # Ruido alto
    spec_noise = AnhadirRuidoEspectral(std_range=(0.08, 0.15)) # Ruido espectral notable
    spec_illum = IluminacionAleatoria(factor_range=(0.4, 0.4)) # Oscurecido fuertemente
    spec_drop = EliminarBandas(drop_prob=0.4) # Apagamos el 40% de las bandas
    
    erasing = v2.RandomErasing(p=1.0, scale=(0.15, 0.15), ratio=(1.0, 1.0), value=0) # Borrado de un cuadrado negro fijo
    
    transformaciones = {
        "0_Original": None,
        "1_Base_Flips": v2.Compose(flips),
        "2_Geometria": v2.Compose(flips + [rotation, simetric_zoom]),
        "3_Oclusion": v2.Compose(flips + [erasing]),
        "4_Ruido_Sensores": v2.Compose(flips + [spec_noise, noise]),
        "5_Firma_Espectral": v2.Compose(flips + [spec_illum, spec_drop]),
        "6_Combo_Equilibrado": v2.Compose(flips + [rotation, simetric_zoom, spec_illum, noise]),
        "7_All_In": v2.Compose(flips + [rotation, simetric_zoom, spec_illum, spec_noise, spec_drop, erasing])
    }
    
    carpeta_salida = "ejemplos_aumentados_raw"
    os.makedirs(carpeta_salida, exist_ok=True)
    
    print("Aplicando aumentados y generando archivos...")
    for nombre, transformacion in transformaciones.items():
        if transformacion is None:
            patch_transformado = patch_original.clone()
        else:
            patch_transformado = transformacion(patch_original)
            
        ruta_salida = os.path.join(carpeta_salida, f"{nombre}.raw")
        # Le pasamos el d_max para que sepa cuánta luz darle
        save_patch(patch_transformado, sizex, sizey, B, ruta_salida)
        
    print(f"\nSe han generado todas las imágenes")