#!/usr/bin/env python3
# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials (pytorch-tutorial-master.zip)
# Adapted to multi/hyperspectral images: F. Arguello
# CNN21: 2 capas convolucionales, 1 completamente conectada
# oitaven WP (15%, texturas+fv+3kelm, t=3m44s): OA=93.03, OA=87.18
# CNN21 SEG EXP: 5 EPOCHS: 100 SAMPLES: [0.15, 0.05] ADA: 3 AUM: 1
# Class 01: 96.66+0.33
# Class 02: 80.72+2.00
# Class 03: 75.76+4.18
# Class 04: 87.01+3.79
# Class 05: 86.18+1.77
# Class 06: 92.11+1.21
# Class 07: 96.46+0.15
# Class 08: 95.77+0.21
# Class 09: 98.66+0.56
# Class 10: 90.81+0.62
# OA=94.77+0.20, AA=90.01+0.66, t=60 s
  
import math, random, struct, signal, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from sklearn import preprocessing
from torchvision.transforms import v2
import torchvision.utils as vutils

from implementations.torchbearer_implementation import FMix

from concurrent.futures import ProcessPoolExecutor

import torch.multiprocessing as mp

import itertools

import sys

import os

import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='multiprocessing.resource_tracker')

EXP=10      # numero de experimentos (NÚMERO DE VECES QUE SE REPITE EL PROCESO DE ENTRENAMIENTO Y PRUEBA), lso resultados serán el promedio de cada resultado
SAMPLES=[0.15,0.05] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05)  (PORCENTAJE DE ENTRENAMIENTO (segmentos usados para entrenar), PORCENTAJE DE VALIDACIÓN (segmentos usados para validar))
ADA=3  # learning rate: 0-fijo, 1-manual, 2-MultiStepLR, 3-CosineAnnealingLR, 4-StepLR
AUM=1  # aumentado: 0-sin_aumentado, 1-con_aumentado
DET=0  # experimentos: 0-aleatorios, 1-deterministas (CON ALEATORIOS SE INICIALIZAN PESOS Y SELECCIÓN DE MUESTRAS AL AZAR)
ALL=0  # testar 0-solo ground-truth, 1-todo

SEMILLA=0



# DATASET='/home/amo/profile.raw'
# GT='/mnt/media/images/salinas_gt.pgm'
# SEG='/home/amo/seg.raw'
# CENTER='/mnt/media/images/seg_salinas_centers.raw'

# DATASET='/mnt/media/images/ermidas_creek.raw'
# GT='/mnt/media/images/ermidas_creek.pgm'
# SEG='/mnt/media/images/seg_ermidas.raw'
# CENTER='/mnt/media/images/seg_ermidas_centers.raw'

#-----------------------------------------------------------------
# FUNCIONES PARA LEER DATASETS Y SELECCIONAR MUESTRAS
#-----------------------------------------------------------------

#Función que se encarga de leer los datos almacenados en el fichero RAW correspondiente al dataset original
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


#Función que se encarga de leer el fichero que contiene la segmentación de los píxeles del dataset(SEG)
def read_seg(fichero):
  #Leemos los dos primeros números presentes en el archivo (los 2 son enteros de 32 bits)
  #H es el ancho de píxeles
  #V es la altura de píxeles
  (H,V)=np.fromfile(fichero,count=2,dtype=np.uint32)
  #Leemos el resto de datos del fichero (H*V enteros de 32 bits) saltando los primeros 8 bytes (los 2 valores de la cabecera de 4 bytes cada uno)
  datos=np.fromfile(fichero,count=H*V,offset=2*4,dtype=np.uint32)
  
  #Se imprime información sobre el fichero de segmentación leído
  #print('Lectura del fichero de segmentación*********')
  #print('* Read segmentation:',fichero)
  #print('  H (anchura):',H,'V (altura):',V)
  #print('  Píxeles leídos:',len(datos))

  #Devolvemos los datos de segmentación junto a la anchura y la altura
  return(datos,H,V)


#Función que permite leer el fichero que contiene los píxeles centrales de los segmentos (CENTER)
def read_seg_centers(fichero):
  #Leemos los 3 primeros números presentes en el archivo (enteros de 32 bits)
  #H es el ancho en píxeles
  #V es el alto en píxeles


  (H,V,nseg)=np.fromfile(fichero,count=3,dtype=np.uint32)
  #Leemos el resto de datos del fichero (H*V enteros de 32 bits) saltando los primeros 12 bytes (los 3 valores de la cabecera)
  datos=np.fromfile(fichero,count=H*V,offset=3*4,dtype=np.uint32)

  #Se imprime información sobre el fichero de centros de segmentos leído
  #print('Lectura del fichero de centros de la segmentación*********')
  #print('* Read centers:',fichero)
  #print('  H (anchura):',H,'V (altura):',V,'nseg (número de segmentos)',nseg)
  #print('  Elementos leídos:',len(datos))

  #Devolvemos los datos sobre los centros de los segmentos junto a al anchura, la altura y el número de segmentos
  return(datos,H,V,nseg)



#Función que se encarga de leer el archivo que contiene los píxeles etiquetados
def read_pgm(fichero):
  #Abrimos el fichero en modo lectura binaria
  try:
    pgmf=open(fichero,"rb")
  except IOError:
    print('No puedo abrir ',fichero)
  else:
    #Se lee la primera línea y se verifica que se corresponda 'P5\n', pues esto indica que el fichero es una imagen en escala de grises 
    assert pgmf.readline().decode()=='P5\n'

    #Se saltan los comentarios del inicio del fichero
    line=pgmf.readline().decode()
    while(line[0]=='#'):
      line=pgmf.readline().decode()
    
    #Se divide la primera línea tras los comentarios en dos partes (H y V) correspondientes al ancho y el alto de la imagen
    (H,V)=line.split()

    #Se transforman los valores en enteros
    H=int(H); V=int(V)

    #Se lee la siguiente línea (correspondiente a la profundidad)
    depth=int(pgmf.readline().decode())

    #Se comprueba que la profundidad sea de 8 bits (menor o igual que 255, es decir puede haber hasta 255 clases)
    assert depth<=255

    #Crea una lista llamada raster donde se meterán los valores byte a byte correspondientes con la etiqueta asociada a cada píxel
    raster=[]
    for i in range(H*V):
      #Se emplea la función ord() para transformar el valor binario en el número entero que representa
      raster.append(ord(pgmf.read(1)))
    
    #print('* Read GT (fichero con los píxeles etiquetados):',fichero)
    #print('  H (anchura):',H,'V (altura):',V,'depth (profundidad de cada valor):',depth)
    #print('  Valores leídos:',len(raster))
    return(raster,H,V)



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

# Esta parte tarda mucho, mejor la preprocesamos en C
#Esta función es la encargada de obtener el punto central de cada segmento de la imagen (es muy lento y no se usa)
def seg_center(seg,H,V):
  #print('* Segment centers (tarda mucho)')
  nseg=0
  #Se comprueba cuantos segmentos hay en total
  for i in range(H*V):
    if(seg[i]>nseg): nseg=seg[i]
  nseg=nseg+1
  #print('  Número de segmentos:',nseg)
  
  #Se crean 4 listas que almacenarán los límites de cada segmento (valor mínimo y máximo de cada segmento en X y en Y)
  xmin=[H*V]*nseg; xmax=[0]*nseg; 
  ymin=[H*V]*nseg; ymax=[0]*nseg; 

  #Se recorren los píxeles comprobando el segmento al que pertenece cada píxel y se actualizan los límites del segmento al que pertenece según sea necesario
  for i in range(H*V):
    x=i%H; y=i//H; s=seg[i]
    if(x<xmin[s]): xmin[s]=x
    if(y<ymin[s]): ymin[s]=y
    if(x>xmax[s]): xmax[s]=x
    if(y>ymax[s]): ymax[s]=y
  
  #Se calcula el centro de los segmentos (de la caja delimitadora que los contiene) en base a los límites de coordenadas de cada uno de ellos
  center=np.zeros(nseg,dtype=np.uint32)
  for s in range(nseg):
    y=(ymin[s]+ymax[s])//2; x=(xmin[s]+xmax[s])//2; 

    #Se convierte la coordenada del centro en un único número empleando el valor de anchura de píxeles
    center[s]=y*H+x
  return(center,nseg)

#Función que decide qué píxeles se emplearán para realizar el entrenamiento, cuales para la validación y cuáles para el test
def select_training_samples_seg(truth,center,H,V,sizex,sizey,porcentaje):
  #print('* Seleccionar elementos para el entrenamiento')
  
  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)
  
  #Recorremos el fichero con las etiquetas para saber cuántas clases hay en total
  for i in truth:
    if(i>nclases): nclases=i
  #print('  nclasses (Número de clases):',nclases)
  lista=[0]*nclases;
  
  #Se crea una lista que almacenará los índices de los píxeles que pertenecen a la categoría asociada a esa lista
  for i in range(nclases):
    lista[i]=[]

  
  #Se calculan los valores mínimos para los centros de los patches (para evitar errores)
  xmin=int(sizex/2); xmax=H-int(math.ceil(sizex/2))
  ymin=int(sizey/2); ymax=V-int(math.ceil(sizey/2))

  #Se recorren los centros de los segmentos
  for ind in center:
    #Convierte el índice plano a coordenadas (y=i, x=j)
    i=ind//H; j=ind%H;

    #Si el centro del segmento se encuentra fuera de los valores permitidos para los centros se descarta el centro
    if(i<ymin or i>ymax or j<xmin or j>xmax): continue

    #Si el centro del segmento está etiquetado en el fichero de ground truth (es decir, tiene un valor mayor a 0) se guarda en la lista de centros correspondiente a su clase
    if(truth[ind]>0): lista[truth[ind]-1].append(ind)

  
  #Se desordena la lista de centros, de esta manera no están ordenados por la disposición de los mismos en la imagen y al tomar los conjuntos se tomaran centros de distintas posiciones dentro de la imagen
  for i in range(nclases):
    random.shuffle(lista[i])
  

  #Seleccionamos las muestras para los conjuntos de entrenamiento, validacion y test
  #print('  Clase  # :   total | train |   val |    test')
  train=[]; val=[]; test=[]
  #
  #Se realiza el reparto de elementos por cada clase
  for i in range(nclases):

    #tot0: numero muestras entrenamiento, tot1: validacion 
    
    #Número muestras de entrenamiento**********************************

    #Si el valor del porcentaje de muestras para entrenamiento es mayor que 1 se asume que es número de elementos en lugar de porcentaje
    if(porcentaje[0]>=1): tot0=porcentaje[0]
    #En caso contrario se asume que se trata de un porcentaje
    else: tot0=int(porcentaje[0]*len(lista[i]))

    #Si se pide un mayor número de muestras de entrenamiento que el número de elementos que hay en el dataset se pasa a mantener únicamente la mitad de elementos para el entrenamiento
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2

    #Nos aseguramos de que siempre haya almenos una muestra de la clase actual para el entrenamiento
    if(tot0<=0 and len(lista[i])>0): tot0=1

    #Si finalmente hay almenos un elemento de la clase actual para el entrenamiento se aumenta el contador de clases no vacías
    if(tot0!=0): nclases_no_vacias+=1


    #Número muestras de validación**********************************

    #Si el valor del porcentaje de muestras para validación es mayor que 1 se asume que es número de elementos en lugar de porcentaje
    if(porcentaje[1]>=1): tot1=porcentaje[1]
    else: tot1=int(porcentaje[1]*len(lista[i]))

    #Si el número de elementos para la validación es mayor o igual al número de elementos que quedan tras eliminar los usados por el entrenamiento se reduce el número de elementos de validación
    #De esta manera se evita que nos quedemos sin elementos para realizar el test
    if(tot1>=len(lista[i])-tot0): tot1=(len(lista[i])-tot0)//2

    #Si no se llegan a tener las suficientes muestras como para tener almenos 1 de validación simplemente se asignan 0 muestras de la clase actual para la validación    
    if(tot1<1 and len(lista[i])>0): tot1=0

    #Se pasan a llenar las listas de entrenamiento validación y test con los centros de la clase actual, que se han desordenado previamente
    for j in range(len(lista[i])):
      #Los primeros centros de la lista de la clase actual van al conjunto de entrenamiento.
      if(j<tot0): train.append(lista[i][j])

      #Los siguientes centros van al conjunto de validación
      elif(j<tot0+tot1): val.append(lista[i][j])

      #En test se incluyen todos los centros, pero a la hora de calcular la precisión del modelo se tendrán en cuenta únicamente los elementos que no se emplearon en el entrenamiento
      test.append(lista[i][j])
    
    #Se imprime un resumen sobre la repartición de los datos en los distintos conjuntos en función de la clase
    #print('  Class',f'{i+1:2d}',':',f'{len(lista[i]):7d}','|',f'{tot0:5d}','|',
      #f'{tot1:5d}','|',f'{len(lista[i])-tot0-tot1:7d}')
    
  return(train,val,test,nclases,nclases_no_vacias)

#Esta función selecciona todos los centros a pesar de que no posean una etiqueta asociada
def select_all_samples_seg(center,H,V,sizex,sizey):
  #print('* Seleccionar todos los centroides')
  #Se calcula el rango de valores admitido para los centroides
  xmin=int(sizex/2); xmax=H-int(math.ceil(sizex/2))
  ymin=int(sizey/2); ymax=V-int(math.ceil(sizey/2))

  #Se recorren todos los centros de los segmentos, si el centroide está dentro de los valores admitidos se añade al conjunto de test
  test=[]
  for ind in center:
    i=ind//H; j=ind%H;
    if(i<ymin or i>ymax or j<xmin or j>xmax): continue
    test.append(ind)
  return(test)

#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------


#Clase asociada al dataset con las etiquetas, igual que el anterior pero con las etiquetas del ground truth
#Usada para el entrenamiento
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey,is_train):
    #Se guarda la imagen (datos), las etiquetas asociadas a los píxeles y los índices de los centros (samples) de los segmentos
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.is_train = is_train

    #Herramienta de aumentado de datos, se realizan estas operaciones con un 50% de probabilidad cada una por separado (es como lanzar varias monedas seguidas)
    #Mediante el aumentado de datos evitamos que cosas como la posiciónd el sol en el momento de la captura de la imagen afecten a la manera de aprender y predecir del modelo una vez entrenado
    self.transform=v2.Compose(
      [v2.RandomHorizontalFlip(),v2.RandomVerticalFlip()])
    
  def __len__(self):
    return len(self.samples)

  #Función que se ejecuta cada vez que se pide un patch para analizar
  def __getitem__(self,idx):
    #Se recuperan los datos almacenados al construir la instancia
    datos=self.datos; truth=self.truth; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 

    #Se convierte el índice del centroide en coordenadas 2D (x e y)
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)

    #Se obtiene el patch alrededor de ese centroide
    patch=select_patch(datos,sizex,sizey,x,y)

    #Si el aumentado de datos está activado se aplican las transformaciones al azar
    if(AUM==1 and self.is_train): 
      patch=self.transform(patch)

    # renumeramos porque la red clasifica tambien la clase 0 
    
    #Se devuelve el patch y la etiqueta restándole 1 debido a que las redes neuronales de PyTorch esperan que las categorías empiecen en 0
    return(patch,truth[self.samples[idx]]-1)

#-----------------------------------------------------------------
# PYTORCH - UTIL
#-----------------------------------------------------------------

# pulsando CNLT-C acabamos el entrenamiento y pasamos a testear

#Signal Handler que permite parar el entrenamiento y pasar a testear el modelo directamente
def signal_handler(sig, frame):
  #print('\n* Ctrl+C. Exit training')
  global endTrain
  endTrain=True

# For updating learning rate manual

#Función empleada para modificar el learning rate durante el entrenamiento para modificar la manera de aprender de la red en las distintas épocas
def update_lr(optimizer,lr):
  #Modificamos el learning rate en todos los grupos de hiperparámetros al nuevo valor recibido por la función
  for param_group in optimizer.param_groups:
    param_group['lr']=lr

# calcula los promedios de precisiones



#-----------------------------------------------------------------
# PYTORCH - NETWORK
#-----------------------------------------------------------------

# Convolutional neural network (two convolutional layers)

#Clase asociada a la red neuronal convolucional
class CNN21(nn.Module):
  #Constructor de la clase
  #N1 = Canales de entrada (profundidad inicial de los datos) (en este caso el número de bandas espectrales de la imagen)
  #N2 = Número de filtros en la primera capa, en esta capa se buscan patrones simples (como gradientes de color)
  #N3 = Número de filtros en la segunda capa, en esta capa se buscan combinaciones de los patrones detectados en la primera capa
  #N4 = Múmero total de características extraídas justo antes de la decisión de clasificación N4 = N3 * ancho final* alto final (datos aplanados), representa todo el conocimiento extraído del patch
  #N5 = El número de clases final (las categorías posibles)
  #D1 y D2 = Parámetros que controlan el desplazamiento de la ventana de pooling en la capa 1 y en la capa 2
  def __init__(self,N1,N2,N3,N4,N5,D1,D2):
    super(CNN21,self).__init__()
    
    #Definimos la primera capa, esta primera capa es el primer filtro de procesamiento por el que pasa cada patch, se realizan las acciones contenidas en Sequential de manera secuencial

    #Con Conv2d se realiza la convolución, pasando una ventana de 3x3 por la imagen, esta ventana se mueve en pasos de 1 píxel (haciendo que el mismo píxel sea analizado más de 1 vez pudiendo ver su relación con los píxeles vecinos), 
    #pasando de N1 bandas a N2 mapas de características donde cada mapa resaltará distintas características de la imagen (por tanto se generan N2 nuevas versiones)
    #Se llama 2d debido a que el filtro solo se desplaza en 2 direcciones, no se desplaza hacia las bandas, el filtro analiza todas las bandas a la vez
    #Se añade un padding de 2 píxeles para que la ventana pueda analizar correctamente los píxeles de los bordes, esto aumenta un poco el tamaño del patch

    #Con BatchNorm2d normalizamos los datos que se originaron en el paso anterior

    #ReLu se trata de las funciones de activación asociadas a cada neurona

    #Con MaxPool2d se realiza una operación de submuestreo, en este caso se comprueba el patch en cuadrados de 2x2 y se conserva el valor más alto, con D1 controlamos cuanto salta la ventana y por tanto cuanto se comprime la imagen


    self.layer1=nn.Sequential(
      nn.Conv2d(N1,N2,kernel_size=3,stride=1,padding=2),
      nn.BatchNorm2d(N2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D1))
    
    #En la capa 2 se reciben los mapas de características de la capa 1 y se realiza el mismo proceso que en la primera capa pero variando los hiperparámetros para detectar relaciones complejas entre las características
    #Aquí la convolución usa una ventana mayor para detectar relaciones más complejas y grandes entre las características
 
    self.layer2=nn.Sequential(
      nn.Conv2d(N2,N3,kernel_size=5,stride=1,padding=2),
      nn.BatchNorm2d(N3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D2))
    
    #Capa final de la clasificación (totalmente conectada)
    #Recibe el vector N4 con todas las características extraídas por las dos capas anteriores y estiradas en una fila
    #Se procesan estas características y se da una puntuación a cada una de las N5 clases posibles, dando lugar a la predicción final (la clase con mayor puntuación)
    self.fc=nn.Linear(N4,N5)
      
  #Función que hace circular los datos entre las capas, x es el patch actual
  def forward(self,x):
    #El patch empieza entrando en la capa 1, donde se aplica la convolución de N1 a N2, la normalización la activación ReLu y el primer pooling (D1)
    #De aquí se obtiene un patch de menor tamaño y con las características básicas resaltadas
    out=self.layer1(x)

    #El resultado de la capa anterior pasa a la segunda capa donde se aplican filtros más grandes para la convolución, se vuelve a normalizar, se vuelve a realizar la activación ReLu y el segundo pooling (D2)
    #De aquí se obtienen los rasgos más abstractos y complejos del patch, comprimidos en un bloque de datos pequeño y muy denso
    out=self.layer2(out)

    #Se convierte el bloque de datos obtenido de la capa 2 y se realiza un aplanado de los datos, pasando de ser un bloque a una fila
    out=out.reshape(out.size(0),-1)

    #Los datos pasan a la capa final, donde se analizan los datos y se le asigna un peso a cada una de las N5 clases
    out=self.fc(out)

    #Se devuelven las puntuaciones de clases asociadas al patch que ha sido analizado
    return out

#-----------------------------------------------------------------
# PYTORCH - MAIN
#-----------------------------------------------------------------

def main(exp, fmix_alpha, fmix_decay, fmix_soft, data_bundle, TEST, EPOCHS, BATCH, probabilidad, usar_sampler,semilla_fija, gpu_id=0):
  #Leemos los datos del data_bundle

  # Datos y dimensiones originales
  datos = data_bundle['datos']
  H, V, B = data_bundle['H'], data_bundle['V'], data_bundle['B']
  
  # Ground Truth y dimensiones (H1, V1)
  truth = data_bundle['truth']
  H1, V1 = data_bundle['H1'], data_bundle['V1']
  
  # Segmentación y dimensiones (H2, V2)
  seg = data_bundle['seg']
  H2, V2 = data_bundle['H2'], data_bundle['V2']
  
  # Centros y dimensiones (H3, V3, nseg)
  center = data_bundle['center']
  H3, V3, nseg = data_bundle['H3'], data_bundle['V3'], data_bundle['nseg']


  #Tomamos la primera referencia temporal antes de realizar el entrenamiento
  time_start=time.time()

  # 1. Device configuration
  #Comprobamos si el sistema tiene una gráfica compatible con CUDA disponible, si es así se pasa a usar la GPU para entrenar y ejecutar el modelo
  cuda=True if torch.cuda.is_available() else False
  #print('* cuda:',cuda)
  
  #Asignamos la GPU según el gpu_id recibido en caso de tener una gpu disponible
  if cuda:
      device = torch.device(f'cuda:{gpu_id}')
  else:
      device = torch.device('cpu')

  #Si la biblioteca cuDNN está disponible se activan las optimizaciones 
  if torch.backends.cudnn.is_available():
    #print('* Activando CUDNN')
    torch.backends.cudnn.enabled=True
    

    torch.backends.cudnn.benchmark=True

  # experimentos deterministas o aleatorios
  #Si semilla_fija posee valor 1 el experimento será determinista
  if(semilla_fija==1):
    #Fijamos la semilla, sumándole exp para que las pruebas en test determinista sean distintas entre sí
    SEED=SEMILLA + exp
    #Establecemos la semilla a 0 para PyTorch, NumPy y Python
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    #Si no se está empleando cuda le indicamos a PyTorch que emplee los algoritmos que den los resultados exactos y establecemos la semilla para el generador de números aleatorios
    if(cuda==False):
      torch.use_deterministic_algorithms(True)
      g=torch.Generator(); g.manual_seed(SEED)
    
    #Si se está empleando la GPU con cuda se le indica a la librería cudnn que use algoritmos deterministas para la convolución
    #También se desactiva el benchmark para seleccionar el mejor algoritmo para evitar usar algoritmos que no sean fijos
    else:
      torch.backends.cudnn.deterministic=True
      torch.backends.cudnn.benchmark=False

  # durante la ejecucion de la red vamos a coger patches de tamano cuadrado

  #Los patches serán de 32x32 píxeles
  sizex=32; sizey=32 

  # 3. Selection training,testing sets



  #Seleccionamos los conjuntos de entrenamiento, validación y test según lo especificado.
  #La función devuelve los índices de los centros de segmentos que van a cada conjunto en base a la proporción indicada para cada conjunto mediante el parámetro SAMPLES

  # Seleccionamos los conjuntos de entrenamiento, validación y test
  (train,val,test,nclases,nclases_no_vacias)=select_training_samples_seg(truth,center,H,V,sizex,sizey,SAMPLES)

  #Creamos el dataset de entrenamiento y el dataset de testeo en base a los conjuntos de entrenamiento y de testeo
  dataset_train=HyperDataset(datos,truth,train,H,V,sizex,sizey, is_train=True)
  #print('  - train dataset:',len(dataset_train))
  dataset_test=HyperDataset(datos,truth,test,H,V,sizex,sizey, is_train=False)
  #print('  - test dataset:',len(dataset_test))

  # Dataloader
  #Número de hilos a usar para el dataloader
  num_workers_dl = 0
  #Indicamos el batch size (cantidad de patches que se van a procesar al mismo tiempo tanto para entrenar como para validar)
  batch_size=BATCH # defecto 100

  sampler=None
  if (usar_sampler==1):
    # 1. Contamos cuántas muestras hay de cada clase en el conjunto de entrenamiento
    class_counts = [0] * nclases
    for ind in train:
      # truth[ind] va de 1 a nclases, restamos 1 para usarlo como índice de la lista
      clase_real = truth[ind] - 1
      class_counts[clase_real] += 1
    
    # 2. Calculamos el peso de cada clase (1 dividido entre la cantidad de muestras)
    class_weights = [1.0/math.sqrt(count) if count > 0 else 0.0 for count in class_counts]

    # 3. Asignamos el peso correspondiente a cada muestra individual
    sample_weights = [0.0] * len(train)
    for i, ind in enumerate(train):
      clase_real = truth[ind] - 1
      sample_weights[i] = class_weights[clase_real]

    # 4. Creamos el Sampler de PyTorch
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    #Con replacement=True se permite repetir muestras minoritarias para rellenar huecos
    sampler = WeightedRandomSampler(
      weights=sample_weights_tensor, 
      num_samples=len(sample_weights_tensor), 
      replacement=True
    )
    #Creamos el dataloader que se usará durante el entrenamiento, sacará los patches del dataset de entrenamiento con el batch size indicado, es decir sacará batch_size patches
    #Con shuffle=True mezclamos los patches que se usan para entrenar (los centros de segmentos), es decir, se meten patches de distintos lugares de la imagen, de esta manera evitamos que el modelo aprenda el orden de los datos
    train_loader=DataLoader(dataset_train,batch_size,sampler=sampler,num_workers=num_workers_dl)
  
  #Si no se ha indicado el uso del aumentado de las clases minoritarias se usa el DataLoader por defecto sin el sampler
  else:
    train_loader=DataLoader(dataset_train,batch_size,shuffle=True, num_workers=num_workers_dl)

  
  #Creamos el dataloader que se usará durante el testeo de la red neuronal
  #En este caso establecemos shuffle=False para poder evaluar correctamente la predicción de la red hecha para cada segmento
  test_loader=DataLoader(dataset_test,batch_size,shuffle=False,num_workers=num_workers_dl)

  # Si queremos validacion
  if(len(val)>0):
    #Creamos el dataset de validación con el conjunto de validación
    dataset_val=HyperDataset(datos,truth,val,H,V,sizex,sizey, is_train=False)
    #print('  - val dataset:',len(dataset_val))
    #Creamos el dataloader que se usará durante la validación de la red neuronal
    #En este caso establecemos shuffle=False para poder evaluar correctamente la predicción de la red hecha para cada segmento
    val_loader=DataLoader(dataset_val,batch_size,shuffle=False)
 
  # 4. Hyper parameters
  #Hiper parámetros de la red
  #lr es el learning rate inicial usado durante el entrenamiento
  #Por defecto está a 0.001
  if(ADA==0): lr=0.001
  else: lr=0.001
 
  # 5. Red: CNN con dos capas convolucionales y una lineal al final
  # 5.1. capa conv.1 (Primera capa convolucional) (Detección básica de relaciones)
  N1=B          # dimension de entrada (bandas)
  D1=2          # decimacion, por defecto 2 (el factor de salto del MaxPool, la ventana de pooling) 
  H1=sizex      # lado patches entrada, por defecto 32 (sizex=sizey) (Tamaño inicial del parche 32x32)
  N2=32         # dimension de salida (seleccionada), por defecto 16. Número de mapas de rasgos
  H2=int(H1/D1) # lado patches salida (calculada), por defecto 16 (sizex=sizey) (El tamaño de los patches al salir (16 x 16))

  # 5.2. capa conv.2, parametros de entrada N2,H2 vienen dados por la capa anterior
  N3=64         # dimension de salida (seleccionada), por defecto 32. Número de mapas de rasgos, en este caso el doble que en la primera capa para buscar muchos más rasgos complejos
  D2=2          # decimacion, por defecto 2 (el factor de salto del MaxPool, la ventana de pooling) 
  H3=int(H2/D2) # lado patches salida (calculada), por defecto 16 (sizex=sizey) (El tamaño de los patches al salir (16 x 16))
    
  # 5.3. capa completamente conectada, parametro de entrada N4 viene de la etapa anterior 
  N4=H3*H3*N3   # dimension de entrada (aplanado de la salida de la segunda capa convolucional)
  N5=nclases    # dimension de salida (número de categorías final)
  
  #Generamos la red y la cargamos en la CPU o GPU (si es compatible con CUDA)
  model=CNN21(N1,N2,N3,N4,N5,D1,D2).to(device)

  # Inicializamos FMix para patches de 32x32 (sizex x sizey)
  fmix_util = FMix(size=(sizex, sizey), alpha=fmix_alpha, decay_power=fmix_decay, max_soft=fmix_soft)

  # 6. Loss, optimizer, and scheduler

  # 6.1 Cross Entropy loss

  #Definimos la función de pérdida que se usará para medir qué tan lejos está la predicción de la red respecto a la realidad
  #En este caso se emplea la entropía cruzada, pues estamos ante un problema de clasificación de múltiples clases
  #La entropía cruzada tiene en cuenta la probabilidad que le dió la red al segmento de ser la clase real, cuanto menor sea la probabilidad dada por la red de ser la clase real mayor es el error
  criterion=nn.CrossEntropyLoss()

  # 6.2 create an optimizer object: Adam optimizer with learning rate lr
  #Especificamos el modelo de optimización que se empleará durante el entrenamiento de la red
  #El modelo de optimización es el encargado de actualizar los pesos de la red para minimizar la función de pérdida
  #En este caso se empleará Adam, el cual es adaptativo y posee un momentum para evitar caer en óptimos locales y otro para adaptar la tasa de aprendizaje de cada parámetro (pesos y sesgos) de forma individual
  optimizer=torch.optim.Adam(model.parameters(),lr=lr)

  # 6.3 scheduler (no es estrictamente necesario)
  #El scheduler es el componente que decide cómo cambiar el learning rate a medida que pasan las épocas de entrenamiento

  #Si ADA==2 se emplea el descenso por escalones fijos, se reduce el learning rate al llegar a la mitad del entrenamiento y casi al final, cuando se llega cada hito se multiplica el lr por el valor de gamma (0.1)
  if(ADA==2): scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[EPOCHS//2,(5*EPOCHS)//6],gamma=0.1)
  
  #Si ADA==3 se emplea el descenso suave por coseno, el lr baja siguiendo la curva de un coseno, empieza en el valor inicial y va bajando a lo largo de todas las épocas
  #Es un descenso suave y continuo
  elif(ADA==3): scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=0)
  
  #Si ADA==4 se emplea el descenso constante y gradual, se reduce el learning rate en cada época, en cada época el lr se multiplica por 0.99
  elif(ADA==4): scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99, verbose=True)
  
  #Si ADA posee otro valor el learning rate pasa a ser fijo durante todo el entrenamiento
  else: pass

  # 7. Train the model
  #Pasamos a realizar el entrenamiento del modelo
  #Configuramos la variable que permite parar el entrenamiento
  global endTrain
  endTrain=False
  # signal.signal(signal.SIGINT,signal_handler)

  #Obtenemos el número de batches que van a ser procesados durante el entrenamiento
  total_step=len(train_loader)


  
  #Tomamos la marca de tiempo inicial
  tiempo_inicial_entrenamiento = time.perf_counter()

  #Bucle de entrenamiento asociado a las épocas
  for epoch in range(EPOCHS):


    
    #Activamos el modo entrenamiento al principio de cada época para que las capas actualicen sus estadísticas internas con cada nuevo época, adaptándose así a los nuevos pesos obtenidos en la época anterior
    model.train()
    
    #Recorremos los patches (inputs) junto a sus etiquetas (labels) del conjunto de entrenamiento, recorremos batches de patches, es decir 1 conjunto de patches en cada iteración, por tanto en cada iteración se procesan BATCH_SIZE patches
    for i,(inputs,labels) in enumerate(train_loader):

      # 7.1. Cogemos muestras para entrenar
      #Cargamos los datos y sus etiquetas en la GPU (o se dejan en la CPU)
      inputs=inputs.to(device)
      labels=labels.to(device)

      #Se aplica FMix en función de si se cumple la probabilidad de aplicarlo
      if(random.random()<probabilidad):
        # Aplicamos FMix a los inputs que ya están en el device (GPU o CPU)
        inputs_mixed = fmix_util(inputs) 
        lam = fmix_util.lam         # Obtenemos el peso de la mezcla
        indices = fmix_util.index   # Obtenemos los índices de las imágenes mezcladas
        
        # 7.2. Forward pass
        #La red procesa los patches y devuelve sus predicciones para cada patch (outputs) usando los patches mezclados
        outputs = model(inputs_mixed)

        #Comparamos las predicciones con las etiquetas reales y se calcula el error.
        # loss = lam * Loss(pred, etiqueta_A) + (1 - lam) * Loss(pred, etiqueta_B)
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[indices])
      
      else:
        #Sin aumentado
        outputs=model(inputs)
        loss=criterion(outputs,labels)
      
      
      # 7.3. Backward and optimize
      # 7.3.1. reset the gradients (PyTorch accumulates gradients on subsequent backward passes)
      #Borramos los gradientes asociados a los errores del lote de patches anterior
      optimizer.zero_grad()
      # 7.3.2. compute accumulated gradients
      #Realizamos la retropropagación, calculando el gradiente (dirección en la que cada peso y sesgo debe moverse para bajar el error)
      loss.backward()
      # 7.3.3. perform parameter update based on current gradients
      #Actualizamos los pesos y sesgos de la red con el optimizador (Adam)
      optimizer.step()
    
    #Aquí ha terminado el entrenamiento asociado a la época ***********************************************************************************

    # si tenemos validacion usamos estas muestras, si no el propio train
    
    #Realizamos la validación y medida final de tiempos en la última época de todas
    if(epoch == EPOCHS - 1):
        #Sincronizamos antes de parar el reloj para asegurar que la GPU terminó la última tarea
      if torch.cuda.is_available():
        torch.cuda.synchronize(device)

      tiempo_final_entrenamiento = time.perf_counter()
      tiempo_total_entrenamiento = tiempo_final_entrenamiento - tiempo_inicial_entrenamiento
      tiempo_epoca_entrenamiento = tiempo_total_entrenamiento / EPOCHS

      if(len(val) > 0):
        model.eval()
        # Inicializamos contadores por clase para la validación
        val_class_correct = [0] * (nclases + 1)
        val_class_total = [0] * (nclases + 1)
          
        with torch.no_grad():
          for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Llenamos los contadores por cada muestra del batch
            for j in range(len(labels)):
              real_class = labels[j].item() + 1 # +1 porque restaste 1 en el Dataset
              val_class_total[real_class] += 1
              if predicted[j] == labels[j]:
                val_class_correct[real_class] += 1
        
        #Calculamos el AA de validación
        val_accuracies = []
        for c in range(1, nclases + 1):
          if val_class_total[c] > 0:
            val_accuracies.append(100 * val_class_correct[c] / val_class_total[c])
        
        current_val_aa = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0
        

    # Decay learning rate (lo decrementamos cconforme aumentan las iteraciones)
    #Si ADA==1 se realiza el descenso manual del learning rate cada 20 épocas, donde se divide el learning rate a la mitad y se le comunica este cambio al optimizer
    if(ADA==1 and (epoch+1)%20==0): lr/=2; update_lr(optimizer,lr)
    #Si está activado el ajuste automático se le informa al scheluder de que ha pasado una época de entrenamiento para que actualice el learning rate según la ecuación seleccionada
    elif(ADA>1): scheduler.step()
    #Si endTrain ha pasado a True se finaliza el entrenamiento
    if(endTrain): break



  #Si no está activado el flag de testeo la función devuelve directamente la media del accuracy asociado al conjunto de validación obtenido en la validación de la última época de entrenamiento
  if(TEST==0): 
    return current_val_aa

  # 8. Test the model
  #print('* Test FINAL SOBRE CONJUNTO DE TEST CNN21, exp.%d'%(exp))
  #Creamos el mapa de salida de clasificación de los píxeles con todo 0
  output=np.zeros(H*V,dtype=np.uint8) # mapa de salida de pixels

  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
  #Ponemos el modelo en modo evaluación
  model.eval()

  #Evitamos que se almacene el grafo asociado a los gradientes empleado durante el entrenamiento, de esta manera aceleramos los cálculos y reducimos el consumo de memoria
  with torch.no_grad():
    correct=0; total=0;

    #Realizamos la clasificación sobre el conjunto de test procesanto batches de patches
    for(inputs,labels) in test_loader:
      #Cargamos los patches
      inputs=inputs.to(device)
      
      #Cargamos las etiquetas asociadas a los patches
      labels=labels.to(device)

      #Realizamos la clasificación de los patches
      outputs=model(inputs)
      
      #Obtenemos la clasificación final asociada a cada patch (la clase con mayor valor establecido por la red)
      (_,predicted)=torch.max(outputs.data,1)

      #Cargamos las predicciones en la RAM
      predicted_cpu=predicted.cpu()

      #Recorremos las predicciones realizadas para cada uno de los patches
      for i in range(len(predicted_cpu)):
        #La red devuelve las clases empezando en 0, por tanto sumamos 1, pues el 0 lo reservamos para zonas sin clasificar o sin datos
        #Se guarda la predicción de cada patch en la posición asociada al centro de segmento que se tomó para realizarlo (la posición del píxel se encuentra en test)
        #Se usa la variable total para llevar la cuenta de los patches que ya han sido procesados, para así estar en la posición correcta del vector de test
        #De esta manera vamos rellenando el mapa de clasificación con los segmentos de test
        output[test[total+i]]=np.uint8(predicted_cpu[i]+1)
      #Aumentamos la variable total con tantas posiciones como patches se hayan procesado en la iteración actual
      total+=labels.size(0)
      #Cada vez que se han clasificado 2000 patches se imprime por pantalla el progreso del testeo
      #if(total%2000==0): #print('  Testeando: %6d/%d'%(total,len(dataset_test)))

  
  
  
  #Tras lo anterior tenemos el mapa con únicamente la clasificación de los píxeles que son centros de segmento, por tanto se debe propagar la clase del centro del segmento a los píxeles del segmento completo
  #print('* Generando mapa de clasificación (only ground-truth) (solo segmentos usados en el testeo)')
  #Recorremos todos los píxeles del output
  #Buscamos a que segmento pertenece el píxel actual seg[i]
  #Buscamos cual es el píxel central del segmento al que pertenece el píxel actual center[seg[i]]
  #Miramos que clase fue asignada al píxel central (mediante el patch) y le asignamos esa clase al píxel actual
  for i in range(H*V): output[i]=output[center[seg[i]]]

  #Eliminamos los centros usados en el entrenamiento y validación de la red, de esta manera en el output todos valdrán 0
  for i in train: output[i]=0
  for i in val: output[i]=0

  
  
  # 9. Calculamos las precisiones por segmentos (excluyendo los usados en el entrenamiento y validación)
  #Contadores para clasificaciones correctas de segmentos y el total de segmentos
  correct=0; total=0
  #Recorremos todos los centros de segmentos que componen la imagen (que cargamos al principio)
  for i in range(len(center)):
    #Si en el output el centro del segmento tiene un valor 0 no lo contabilizamos (pues fue usado en el entrenamiento o en la validación)
    if(output[center[i]]==0): continue
    #En caso contrario se aumenta el contador de segmentos totales (pues un centro se corresponde con un segmento)
    total+=1
    #Se comprueba lo que al red predijo para el segmento, si es igual a la clasificación real del segmento se aumenta el contador de clasificaciones correctas
    if(output[center[i]]==truth[center[i]]): correct=correct+1
  #Calculamos el accuracy 
  acc=100*correct/total;
  #print('* Accuracy (Overall Accuracy) a nivel de segmentos: %.02f'%(acc))

  # 10. precisiones a nivel de pixel
  #Realizamos el cálculo de la precisión a nivel de píxel
  
  #Creamos los contadores y listas necesarias para calcular el overall accuracy y el average accuracy
  correct=0; total=0; AA=0; OA=0
  #Se suma 1 al número de clases debido a que se debe añadir la clase 0
  class_correct=[0]*(nclases+1)
  class_total=[0]*(nclases+1)
  class_aa=[0]*(nclases+1)

  #Recorremos los píxeles que fueron clasificados por la red (almacenados en output)
  for i in range(len(output)):
    #Si el píxel fue empleado en entrenamiento y/o validación o bien no está clasificado en el mapa de etiquetas original se ignora
    if(output[i]==0 or truth[i]==0): continue

    #En caso contrario se aumenta el contador de píxeles total del conjunto de test y el contador de píxeles de la clase a la que pertenezca el píxel realmente 
    total+=1; class_total[truth[i]]+=1
    
    #Si la predicción de red coincide con la etiqueta real del píxel se aumentan los contadores de éxitos globales y el de la clase asociada al píxel
    if(output[i]==truth[i]):
      correct+=1
      class_correct[truth[i]]+=1
  
  #Ahora pasamos a calcular el accuracy de cada clase
  for i in range(1,nclases+1):
    #Si existen píxeles de la clase actual en el conjunto de test se pasa a calcular el accuracy asociado a la misma
    if(class_total[i]!=0): class_aa[i]=100*class_correct[i]/class_total[i]
    #Si no existen píxeles de la clase actual en el conjunto de test se asigna un 0% de accuracy
    else: class_aa[i]=0
    #Suma el accuracy de la clase actual al resto de accuracys asociados a las otras clases
    AA+=class_aa[i]
  
  #Calculamos el Overall Accuracy y el Average Accuracy (dividiendo entre nclase_no_vacias para no dividir por clases que no están en el conjunto de test)
  OA=100*correct/total; AA=AA/nclases_no_vacias 
  

  #Tomamos la referencia de tiempo final del experimento completo
  time_end=time.time()

  print("ACABÓ LA PRUEBA")

  #Finalizamos el main Devolviendo el Overall Accuracy del modelo, el Average Accuracy y el accuracy asociado a cada clase presente en el conjunto de test
  return( OA, AA, class_aa, class_total, tiempo_total_entrenamiento, tiempo_epoca_entrenamiento)



#Función que permite ejecutar el entrenamiento y prueba de la cnn con un conjunto de hiperparámetros determinado
def run_combination(params_with_data):
    torch.set_num_threads(1)
    gpu_id, params, data_bundle = params_with_data
    a, d, s, e, b, p, samp= params
    
    val_acc_list = []

    print(f"[GPU: {gpu_id}] Evaluando: Alpha={a}, Decay={d}, Soft={s}, Epochs={e}, Batch={b}, Prob={p}, Usar_sampler={samp}")
    sys.stdout.flush()

    #Se ejecuta la prueba una única vez
    for exp in range(1):
        res = main(exp, a, d, s, data_bundle,0, e, b, p, samp, 1, gpu_id)
        # Maneja si main devuelve una tupla o un solo valor según el valor de TEST
        v_acc = res[0] if isinstance(res, tuple) else res
        val_acc_list.append(v_acc)

    print(f"[GPU: {gpu_id}]  Fin evaluación: Alpha={a}, Decay={d}, Soft={s}, Epochs={e}, Batch={b}, Prob={p}, Usar_sampler={samp} *************************")
    sys.stdout.flush()
    
    return {'alpha': a, 'decay': d, 'soft': s, 'epochs':e,'batch':b ,'prob':p,'sampler':samp,'mean_val_aa': np.mean(val_acc_list)}


#Función que permite ejecutar el entrenamiento y testeo final del modelo empleando los hiperparámetros óptimos
def run_final_eval(args):
    torch.set_num_threads(1)
    gpu_id, exp_idx, alpha, decay, soft, epochs, batch, prob, samp, data_bundle = args
    oa, aa, class_aa, class_total, tiempo_total_entrenamiento, tiempo_epoca = main(exp_idx, alpha, decay, soft, data_bundle, 1, epochs, batch, prob, samp, DET,gpu_id)
    return oa, aa, class_aa, class_total, tiempo_total_entrenamiento, tiempo_epoca


#Si se lanza el fichero directamente se entra en el entrenamiento y validación
if __name__ == '__main__':
    #Hacemos que los procesos hijo sean totalmente independientes
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    #Detectar cuántas GPUs hay disponibles
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"GPUs detectadas: {num_gpus}")
    
    directorio_actual=os.path.dirname(os.path.abspath(__file__))

    directorio_datos=os.path.join(directorio_actual,'..','datosEntrada')
    

    #Si no se ha indicado un número asociado a un dataset se ejecuta la prueba asociada al dataset del río Oitaven
    if len(sys.argv)<3:
      ficheroLeido="oitaven"
      try:
        usar_sampler=int(sys.argv[1])
      except ValueError:
          print("Error: El argumento debe ser un número entero.")
          sys.exit(1)

      print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
      #Dataset: dataset original, contiene la información obtenida por el dron (cada píxel tiene un cierto número de bandas con datos en cada una)
      DATASET= os.path.join(directorio_datos, 'oitaven', 'oitaven_river.raw')
      #GT: Etiquetas de cada segmento, son las etiquetas reales correspondientes a cada segmento, los segmentos son de 32 x 32 píxeles centrados en un centro.
      GT= os.path.join(directorio_datos, 'oitaven', 'oitaven_river.pgm')
      #SEG: segmentación, cada píxel tiene el ID del segmento al que pertenece
      SEG= os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp.raw')
      #CENTER: centros de los segmentos, contiene los índices de cada píxel correspondiente al centro de cada segmento.
      CENTER= os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp_centers.raw')
    
    else:
      try:
        opcion = int(sys.argv[1])
        usar_sampler=int(sys.argv[2])
      except ValueError:
          print("Error: El argumento debe ser un número entero.")
          sys.exit(1)
      
      match opcion:
        case 1:
          ficheroLeido="das_mestas"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'das_mestas', 'das_mestas_river.raw')
          GT= os.path.join(directorio_datos, 'das_mestas', 'das_mestas_river.pgm')
          SEG= os.path.join(directorio_datos, 'das_mestas', 'seg_mestas_wp.raw')
          CENTER= os.path.join(directorio_datos, 'das_mestas', 'seg_mestas_wp_centers.raw')
        case 2:
          ficheroLeido="eiras_dam"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'eiras_dam', 'eiras_dam.raw')
          GT= os.path.join(directorio_datos, 'eiras_dam', 'eiras_dam.pgm')
          SEG= os.path.join(directorio_datos, 'eiras_dam', 'seg_eiras_wp.raw')
          CENTER= os.path.join(directorio_datos, 'eiras_dam', 'seg_eiras_wp_centers.raw')
        case 3:
          ficheroLeido="ermidas_creek"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'ermidas_creek', 'ermidas_creek.raw')
          GT= os.path.join(directorio_datos, 'ermidas_creek', 'ermidas_creek.pgm')
          SEG= os.path.join(directorio_datos, 'ermidas_creek', 'seg_ermidas_wp.raw')
          CENTER= os.path.join(directorio_datos, 'ermidas_creek', 'seg_ermidas_wp_centers.raw')
        case 4:
          ficheroLeido="ferreiras_river"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'ferreiras_river', 'ferreiras_river.raw')
          GT= os.path.join(directorio_datos, 'ferreiras_river', 'ferreiras_river.pgm')
          SEG= os.path.join(directorio_datos, 'ferreiras_river', 'seg_ferreiras_wp.raw')
          CENTER= os.path.join(directorio_datos, 'ferreiras_river', 'seg_ferreiras_wp_centers.raw')
        case 5:
          ficheroLeido="mera_river"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'mera_river', 'mera_river.raw')
          GT= os.path.join(directorio_datos, 'mera_river', 'mera_river.pgm')
          SEG= os.path.join(directorio_datos, 'mera_river', 'seg_mera_wp.raw')
          CENTER= os.path.join(directorio_datos, 'mera_river', 'seg_mera_wp_centers.raw')
        case 6:
          ficheroLeido="ulla"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET = os.path.join(directorio_datos, 'ulla', 'ulla_river.raw')
          GT= os.path.join(directorio_datos, 'ulla', 'ulla_river.pgm')
          SEG= os.path.join(directorio_datos, 'ulla', 'seg_ulla_wp.raw')
          CENTER= os.path.join(directorio_datos, 'ulla', 'seg_ulla_wp_centers.raw')
        case 7:
          ficheroLeido="xesta"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'xesta', 'xesta_basin.raw')
          GT= os.path.join(directorio_datos, 'xesta', 'xesta_basin.pgm')
          SEG= os.path.join(directorio_datos, 'xesta', 'seg_xesta_wp.raw')
          CENTER= os.path.join(directorio_datos, 'xesta', 'seg_xesta_wp_centers.raw')
        case _:
          ficheroLeido="oitaven"
          print("********************Ejecutando prueba sobre el dataset "+ficheroLeido+ " ******************************")
          DATASET= os.path.join(directorio_datos, 'oitaven', 'oitaven_river.raw')
          GT= os.path.join(directorio_datos, 'oitaven', 'oitaven_river.pgm')
          SEG= os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp.raw')
          CENTER= os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp_centers.raw')
      

    
    #Cargamos los datos una única vez (serán compartidos por los procesos hijo)
    print("Cargando datos en memoria principal...")
    (datos_raw, H, V, B) = read_raw(DATASET)
    (truth, H1, V1) = read_pgm(GT)
    (seg, H2, V2) = read_seg(SEG)
    (center, H3, V3, nseg) = read_seg_centers(CENTER)

    #Reordenamos y preparamos el tensor, dejándolo ordenado en memoria
    datos_tensor = datos_raw.permute(2, 0, 1).contiguous()

    #Liberamos la memoria asociada a los datos leídos y que ya fueron copiados y transformados en la línea anterior
    del datos_raw

    #Hacemos que los datos raw (el dataset original) sean compartidos por todos los procesos hijo, evitando que se copien para cada proceso hijo
    datos_tensor.share_memory_()

    #Creamos el bundle
    data_bundle = {
        'datos': datos_tensor,
        'H': H, 'V': V, 'B': B,
        'truth': truth, 'H1': H1, 'V1': V1,
        'seg': seg, 'H2': H2, 'V2': V2,
        'center': center, 'H3': H3, 'V3': V3,
        'nseg': nseg
    }


    #Archivo que contiene la mejor configuración de hiperparámetros en función de si está usando el sampler o no (el aumentado de clases minoritarias mediante el data loader)
    if(usar_sampler==1):
      
      archivoParametros = "hiperParametros_FMIX_Con_Aumentado.json"
    else:
      archivoParametros = "hiperParametros_FMIX.json"

    mejor_config=None
    #Si existe el fichero que contiene los hiperparámetros optimizados pasamos a abrirlo y cargar los hiperparámetros optimizados
    if os.path.exists(archivoParametros):
      print(f"--- Cargando hiperparámetros óptimos desde {archivoParametros} ---")
      with open(archivoParametros, 'r') as f:
        mejor_config = json.load(f)

    #Si no existe el fichero que contiene los hiperparámetros optimizados y nos encontramos ante el dataset oitaven pasamos a optimizarlos
    if mejor_config is None:
      if ficheroLeido!="oitaven":
        print("ERROR: No hay parámetros optimizados almacenados")
        sys.exit(1)
      else:
        #CONFIGURACIÓN DEL RANDOM SEARCH
        alphas =  [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
        decays =  [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
        softs  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        epochs= [100]
        batches = [256]
        probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        sampler=[usar_sampler]


        PRUEBAS = 200

        if(DET==1 ):
          random.seed(SEMILLA)


        combinaciones_totales = list(itertools.product(alphas, decays, softs, epochs, batches, probs, sampler))

        # 2. Seleccionamos un número máximo de combinaciones al azar
        if len(combinaciones_totales) > PRUEBAS:
          combinaciones = random.sample(combinaciones_totales, PRUEBAS)
        else:
          combinaciones = combinaciones_totales

        tareas = [(i % num_gpus, comb, data_bundle) for i, comb in enumerate(combinaciones)]
        
        print(f"--- Iniciando Random Search Paralelo ({len(combinaciones)} de {len(combinaciones_totales)} combinaciones totales) ---")

        #Ejecutamos el grid search con 5 procesos
        with ProcessPoolExecutor(max_workers=4) as executor:
            resultados_finales = list(executor.map(run_combination, tareas))
            executor.shutdown(wait=True)
        
        time.sleep(0.5)

        #RESULTADOS DEL GRID SEARCH
        resultados_finales.sort(key=lambda x: x['mean_val_aa'], reverse=True)
        mejor_config = resultados_finales[0]

        #Almacenamos los hiperparámetros optimizados
        with open(archivoParametros,'w') as f:
          json.dump(mejor_config,f)


    #EVALUACIÓN FINAL PARALELIZADA
    print(f"\n--- Ejecutando evaluación final paralela ({EXP} experimentos) ---")
    
    
    #Especificamos los parámetros asociados a la mejor configuración
    tareas_finales = [
        (i%num_gpus, i, mejor_config['alpha'], mejor_config['decay'], mejor_config['soft'],mejor_config['epochs'],mejor_config['batch'], mejor_config['prob'], mejor_config['sampler'],data_bundle) 
        for i in range(EXP)
    ]
    
    #Ejecutamos el test final con 4 procesos
    with ProcessPoolExecutor(max_workers=4) as executor:
        resultados_test = list(executor.map(run_final_eval, tareas_finales))
        executor.shutdown(wait=True)
    
    time.sleep(0.5)

    #EXTRACCIÓN Y CÁLCULO DE ESTADÍSTICAS
    final_oa_list = [res[0] for res in resultados_test]
    final_aa_list = [res[1] for res in resultados_test]
    class_aa_matrix = np.array([res[2] for res in resultados_test])
    class_total_matrix = np.array([res[3] for res in resultados_test])

    #Calculamos la media de muestras por clase usadas en los test
    m_total = np.mean(class_total_matrix, axis=0)

    #Listas para almacenar los tiempo de entrenamiento totales y los tiempos por época para cada test
    final_tiempo_total_list = [res[4] for res in resultados_test]
    final_tiempo_epoch_list = [res[5] for res in resultados_test]

    m_oa, s_oa = np.mean(final_oa_list), np.std(final_oa_list, ddof=1)
    m_aa, s_aa = np.mean(final_aa_list), np.std(final_aa_list, ddof=1)

    #Calculamos el tiempo total medio de entrenamiento y el tiempo medio por época de entrenamiento
    m_t_total = np.mean(final_tiempo_total_list)
    m_t_epoch = np.mean(final_tiempo_epoch_list)


    m_class = np.mean(class_aa_matrix, axis=0)
    s_class = np.std(class_aa_matrix, axis=0, ddof=1)

    #IMPRESIÓN DE RESULTADOS FINALES
    print("\n" + "="*60)
    print("RESULTADOS FINALES PROMEDIADOS (CONJUNTO DE TEST) SOBRE EL FICHERO: "+ ficheroLeido)
    print("="*60)
    print(f"Mejor Configuración: Alpha={mejor_config['alpha']}, Decay={mejor_config['decay']}, Soft={mejor_config['soft']}, Epoch={mejor_config['epochs']}, Batch={mejor_config['batch']}, Prob={mejor_config['prob']}, Sampler={mejor_config['sampler']}")
    print("-" * 60)
    
    print(f"ACCURACY POR CLASE:")
    for j in range(1, len(m_class)): 
      #Si la media de muestras de la clase usadas en test es mayor que 0, la clase existía en el test
      if m_total[j] > 0: 
          print(f"  Clase {j:02d}: {m_class[j]:.2f}% ± {s_class[j]:.2f}%")

    print("-" * 60)
    print(f"OA Final: {m_oa:.2f}% ± {s_oa:.2f}%")
    print(f"AA Final: {m_aa:.2f}% ± {s_aa:.2f}%")

    print("-" * 60)

    print(f"Tiempo entrenamiento total medio con hiperparámetros óptimos: {m_t_total:.2f} s")
    print(f"Tiempo medio por época con hiperparámetros óptimos: {m_t_epoch:.4f} s")

    print("="*60)

    
    
