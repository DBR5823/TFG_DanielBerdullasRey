#!/usr/bin/env python3
import math
import os
import numpy as np
import torch
import torchvision.transforms.v2 as v2

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        # Sumamos el ruido
        noisy_tensor = tensor + torch.randn(tensor.size()) * self.std + self.mean
        # IMPORTANTE: Forzamos a que los valores se mantengan en el rango [0, 1]
        return torch.clamp(noisy_tensor, 0.0, 1.0)

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
    
    print("\n--> Buscando un parche brillante y con contenido real...")
    x_centro, y_centro = None, None
    
    for y in range(sizey, V - sizey, 20):
        for x in range(sizex, H - sizex, 20):
            patch_prueba = select_patch(datos_tensor, sizex, sizey, x, y)
            if patch_prueba.mean() > 0.5:
                x_centro = x
                y_centro = y
                break
        if x_centro is not None:
            break
            
    print(f"--> ¡Parche válido encontrado en X={x_centro}, Y={y_centro}!")
    patch_original = select_patch(datos_tensor, sizex, sizey, x_centro, y_centro)
    
    transformaciones = {
        "1_Original": None,
        "2_Rotacion_45": v2.RandomRotation(degrees=(45, 45)), 
        "3_Resized_Crop": v2.RandomResizedCrop(size=(sizex, sizey), scale=(0.6, 0.6), antialias=True),
        "4_Ruido_Gaussiano": AddGaussianNoise(std=0.1) 
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
        
    print(f"\n¡Proceso terminado! Abre los nuevos ficheros, que esta vez sí vas a ver la luz.")