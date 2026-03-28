#!/usr/bin/env python3
# https://github.com/BobMcDear/vit-pytorch
# ViT DE LA LIBRERIA TIMM
# ViT SEG EXP: 5 EPOCHS: 100 SAMPLES: [0.15, 0.05] AUM: 1
# Means and deviations (5 exp):
# * ViT
# Class 01: 96.21+0.42
# Class 02: 84.82+1.39
# Class 03: 80.34+1.92
# Class 04: 92.22+2.97
# Class 05: 88.92+2.91
# Class 06: 96.80+1.67
# Class 07: 97.09+0.27
# Class 08: 96.51+0.42
# Class 09: 99.63+0.11
# Class 10: 94.84+0.57
# OA=96.05+0.21, AA=92.74+0.59
# * SWIN
# Class 01: 95.79+1.04
# Class 02: 79.80+1.40
# Class 03: 76.95+2.26
# Class 04: 91.53+1.32
# Class 05: 83.58+2.17
# Class 06: 93.53+2.24
# Class 07: 96.42+0.52
# Class 08: 95.38+0.28
# Class 09: 98.64+0.43
# Class 10: 92.17+0.62
# OA=94.75+0.21, AA=90.38+0.36

import math, random, struct, signal, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, WeightedRandomSampler
from sklearn import preprocessing
import torchvision.transforms as transforms
import sklearn.utils.class_weight as class_weight
import torch.nn.functional as F

import timm

from torchvision.transforms import v2

from concurrent.futures import ProcessPoolExecutor
from torchvision.transforms import InterpolationMode
import torch.multiprocessing as mp
import sys, os, json, itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='multiprocessing.resource_tracker')

from implementations.torchbearer_implementation import FMix

EXP=10      # numero de experimentos (NÚMERO DE VECES QUE SE REPITE EL PROCESO DE ENTRENAMIENTO Y PRUEBA), los resultados serán el promedio de cada resultado
SAMPLES=[0.15,0.05] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05) (PORCENTAJE DE ENTRENAMIENTO (segmentos usados para entrenar), PORCENTAJE DE VALIDACIÓN (segmentos usados para validar))
EPOCHS=100 # EPOCHS de entrenamiente del clasificador (defecto=100)  **[NO CAMBIES ESTO]**
BATCH=256  # batch-size, defecto=100 
SIZEX=32   # tamano del patch (defecto=32)  **[NO CAMBIES ESTO]**
DET=0      # experimentos: 0-aleatorios, 1-deterministas (defecto=0)
AUM=1      # aumentado: 0-sin_aumentado, 1-con_aumentado (defecto=1)

SEMILLA=0


ViTsize=4  # 0-micro, 1-mini, 2-base, 3-large, 4-pruebas (Tamaño y complejidad del modelo Transformer a utilizar)
ViTtype=0  # 0-vit, 1-swin (Tipo de arquitectura: 0 para Vision Transformer clásico, 1 para Swin Transformer)
TEST=1     # 0-validacion, 1-test
	   

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
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)

  # esta red no necesita realmente normalizar
  #Aplicamos escalado Min-Max con rango [0,1]
  datos=preprocessing.minmax_scale(datos)

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
  #Devolvemos los datos de segmentación junto a la anchura y la altura
  return(datos,H,V)



#Función que permite leer el fichero que contiene los píxeles centrales de los segmentos (CENTER)
def read_seg_centers(fichero):
  #Leemos los 3 primeros números presentes en el archivo (enteros de 32 bits)
  #H es el ancho en píxeles
  #V es el alto en píxeles
  #nseg es el número total de segmentos (aunque no se use directamente aquí, viene en la cabecera)
  (H,V,nseg)=np.fromfile(fichero,count=3,dtype=np.uint32)
  #Leemos el resto de datos del fichero (H*V enteros de 32 bits) saltando los primeros 12 bytes (los 3 valores de la cabecera)
  datos=np.fromfile(fichero,count=H*V,offset=3*4,dtype=np.uint32)

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
    #Creamos una lista llamada raster donde se meterán los valores byte a byte correspondientes con la etiqueta asociada a cada píxel
    raster=[]
    for i in range(H*V):
      #Se emplea la función ord() para transformar el valor binario en el número entero que representa
      raster.append(ord(pgmf.read(1)))
    return(raster,H,V)



#Función que obtiene un patch a partir de un píxel concreto (un centroide de un segmento), crea el patch alrededor del píxel dado.
#Sizex y sizey establecen el tamaño del patch
#x e y son las coordenadas del píxel central
def select_patch(datos,sizex,sizey,x,y):
  #X1 e Y1 son la parte superior izquierda del patch
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));
  #X2 e Y2 son la parte inferior derecha del patch     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  #Se realiza el slice sobre los datos, obteniendo el patch
  patch=datos[:,y1:y2,x1:x2]
  return(patch)







#Función que decide qué píxeles se emplearán para realizar el entrenamiento, cuales para la validación y cuáles para el test
def select_training_samples_seg(truth,center,H,V,sizex,sizey,porcentaje):

  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)

  #Recorremos el fichero con las etiquetas para saber cuántas clases hay en total
  for i in truth:
    if(i>nclases): nclases=i
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
  
  # seleccionamos muestras para train, validacion y test
  train=[]; val=[]; test=[]
  for i in range(nclases):
    # tot0: numero muestras entrenamiento, tot1: validacion
    
    #Número muestras de entrenamiento**********************************
    #Si el valor del porcentaje de muestras para entrenamiento es mayor que 1 se asume que es número de elementos en lugar de porcentaje 
    if(porcentaje[0]>=1): tot0=porcentaje[0]
    #En caso contrario se asume que se trata de un porcentaje
    else: tot0=int(porcentaje[0]*len(lista[i]))

    #Si se pide un mayor número de muestras de entrenamiento que el número de elementos que hay en el dataset se pasa a mantener únicamente la mitad de elementos para el entrenamiento
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2
    #Nos aseguramos de que siempre haya almenos una muestra de la clase actual para el entrenamiento
    if(tot0<0 and len(lista[i])>0): tot0=1
    
    #Si finalmente hay almenos un elemento de la clase actual para el entrenamiento se aumenta el contador de clases no vacías
    if(tot0!=0): nclases_no_vacias+=1

    #Número muestras de validación**********************************
    #Si el valor del porcentaje de muestras para validación es mayor que 1 se asume que es número de elementos en lugar de porcentaje
    if(porcentaje[1]>=1): tot1=porcentaje[1]
    else: tot1=int(porcentaje[1]*len(lista[i]))

    #Si el número de elementos para la validación es mayor o igual al número de elementos que quedan tras eliminar los usados por el entrenamiento se reduce el número de elementos de validación
    if(tot1>=len(lista[i])): tot1=len(lista[i])//2
    
    #Si no se llegan a tener las suficientes muestras como para tener almenos 1 de validación simplemente se asignan 0 muestras de la clase actual para la validación 
    if(tot1<1 and len(lista[i])>0): tot1=0

    #Se pasan a llenar las listas de entrenamiento validación y test con los centros de la clase actual, que se han desordenado previamente
    for j in range(len(lista[i])):
      #Los primeros centros de la lista de la clase actual van al conjunto de entrenamiento.
      if(j<tot0): train.append(lista[i][j])
      #Los siguientes centros van al conjunto de validación
      elif(j<tot0+tot1): val.append(lista[i][j])
      # testeamos los segmentos que no esten en el entrenamiento
      # else: test.append(lista[i][j])
      # testeamos todos los segmentos
      #En test se incluyen todos los centros, pero a la hora de calcular la precisión del modelo se tendrán en cuenta únicamente los elementos que no se emplearon en el entrenamiento
      test.append(lista[i][j])
  return(train,val,test,nclases,nclases_no_vacias)

#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------


# cogemos muestras con ground-truth (dadas por el indice samples)
#Clase asociada al dataset con las etiquetas, igual que el anterior pero con las etiquetas del ground truth. Usada para el entrenamiento.
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey, is_train):
    #Se guarda la imagen (datos), las etiquetas asociadas a los píxeles y los índices de los centros (samples) de los segmentos
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.is_train=is_train

    #Herramienta de aumentado de datos, se realizan estas operaciones con un 50% de probabilidad cada una por separado (es como lanzar varias monedas seguidas)
    #Mediante el aumentado de datos evitamos que cosas como la posiciónd el sol en el momento de la captura de la imagen afecten a la manera de aprender y predecir del modelo una vez entrenado
    flips = [v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)]
    
    t_list = flips.copy()

    #Rotaciones
    # Calculamos un padding suficiente para que al rotar 32x32 no queden huecos
    # La diagonal de 32x32 es aprox 45. Un padding de 8 a cada lado nos da 48x48.
    # Envolvemos en RandomApply para asegurar probabilidad de 0.5
    rotation = v2.RandomApply([v2.Compose([
        v2.Pad(padding=8, padding_mode='reflect'),
        v2.RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.NEAREST),
        v2.CenterCrop(size=(self.sizey, self.sizex))
    ])], p=0.5)

    #Zoom in y zoom out
    # Envolvemos en RandomApply para asegurar probabilidad de 0.5
    simetric_zoom = v2.RandomApply([v2.RandomAffine(degrees=0, scale=(0.8, 1.2))], p=0.5)

    #Eliminación de zonas aleatorias del patch (se eliminan los datos en todas las bandas)
    erasing = v2.RandomErasing(p=0.5, scale=(0.01, 0.05), value=0)



    # Rotación + Zoom Simétrico + Borrado Aleatorio (sobre flips)
    t_list.extend([rotation, simetric_zoom, erasing])

    self.transform = v2.Compose(t_list)
    
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
    if(AUM==1 and self.is_train): patch=self.transform(patch)
    # renumeramos porque la red clasifica tambien la clase 0 
    #Se devuelve el patch y la etiqueta restándole 1 debido a que las redes neuronales de PyTorch esperan que las categorías empiecen en 0
    return(patch,truth[self.samples[idx]]-1)

#-----------------------------------------------------------------
# PYTORCH - UTIL
#-----------------------------------------------------------------

# pulsando CNLT-C acabamos el entrenamiento y pasamos a testear
#Signal Handler que permite parar el entrenamiento y pasar a testear el modelo directamente
def signal_handler(sig, frame):
  print('\n* Ctrl+C. Exit training')
  global endTrain
  endTrain=True



# For updating learning rate manual
#Función empleada para modificar el learning rate durante el entrenamiento para modificar la manera de aprender de la red en las distintas épocas
def update_lr(optimizer,lr):
  #Modificamos el learning rate en todos los grupos de hiperparámetros al nuevo valor recibido por la función    
  for param_group in optimizer.param_groups:
    param_group['lr']=lr


def aplicar_cutmix(inputs, labels, alpha):
  '''Corta un rectángulo de una imagen y lo pega en otra'''
  lam = np.random.beta(alpha, alpha)
  index = torch.randperm(inputs.size(0), device=inputs.device)

  # Calcular coordenadas del cuadro
  W, H = inputs.size(2), inputs.size(3)
  cut_rat = np.sqrt(1. - lam)
  cut_w = int(W * cut_rat)
  cut_h = int(H * cut_rat)
  cx = np.random.randint(W)
  cy = np.random.randint(H)

  bbx1 = np.clip(cx - cut_w // 2, 0, W)
  bby1 = np.clip(cy - cut_h // 2, 0, H)
  bbx2 = np.clip(cx + cut_w // 2, 0, W)
  bby2 = np.clip(cy + cut_h // 2, 0, H)

  # Clonar el tensor para no destruir los datos originales
  inputs_mixed = inputs.clone()

  # Pegar el parche en el tensor CLONADO, sacando la información del tensor ORIGINAL
  inputs_mixed[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]

  # Ajustar lambda según el área real cortada
  lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

  # Devolver el tensor modificado, dejando "inputs" intacto
  return inputs_mixed, labels, labels[index], lam




# selecciona la funcion de perdida
# Esta función permite cambiar dinámicamente el tipo de pérdida empleada en el entrenamiento, útil para datasets desbalanceados
def select_loss(str_loss, truth, device, n_classes):
  #Pérdida de Entropía Cruzada estándar
  if str_loss == 'CE' or str_loss == 'ce':
    loss=nn.CrossEntropyLoss()
    return loss

  #Entropía Cruzada Balanceada: asigna más peso a las clases con menos muestras
  if str_loss == 'balanced_CE' or str_loss == 'balanced_ce':
    #Filtramos los píxeles sin clasificar
    truth_no_zeros=[x for x in truth if x != 0]
    truth_no_zeros=np.array(truth_no_zeros)
    all_classes=np.array(range(1, n_classes+1))
    class_weights=np.ones_like(all_classes, dtype=np.float32)
    unique_classes=np.unique(truth_no_zeros)
    #Calculamos los pesos inversamente proporcionales a la frecuencia de las clases
    calculated_weights=class_weight.compute_class_weight(class_weight='balanced',
       classes=np.unique(truth_no_zeros),y=truth_no_zeros)
    class_weights[np.isin(all_classes, unique_classes)]=calculated_weights
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    #Movemos los pesos a la GPU/CPU
    class_weights=class_weights.to(device)
    loss=nn.CrossEntropyLoss(weight=class_weights)
    return loss
  
  #Pérdida Focal: se centra más en los ejemplos difíciles de clasificar
  if str_loss == 'focal_class':
    loss=FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
    return loss

#-----------------------------------------------------------------
# PYTORCH - MAIN
#-----------------------------------------------------------------

def main(exp, fmix_alpha, fmix_decay, fmix_soft, cutmix_alpha, data_bundle, TEST, EPOCHS, BATCH, probabilidad, probabilidad2, usar_sampler, semilla_fija, gpu_id=0):
  
  # Desempaquetado del data_bundle
  datos = data_bundle['datos']
  H, V, B = data_bundle['H'], data_bundle['V'], data_bundle['B']
  truth = data_bundle['truth']
  H1, V1 = data_bundle['H1'], data_bundle['V1']
  seg = data_bundle['seg']
  H2, V2 = data_bundle['H2'], data_bundle['V2']
  center = data_bundle['center']
  H3, V3, nseg = data_bundle['H3'], data_bundle['V3'], data_bundle['nseg']


  #Tomamos la primera referencia temporal antes de realizar el entrenamiento
  time_start=time.time()


  # 1. Device configuration
  #Comprobamos si el sistema tiene una gráfica compatible con CUDA disponible, si es así se pasa a usar la GPU para entrenar y ejecutar el modelo
  cuda=True if torch.cuda.is_available() else False
  #print('* Cuda: '+str(cuda))
  #Asignamos la GPU según el gpu_id recibido en caso de tener una gpu disponible
  if cuda:
      device = torch.device(f'cuda:{gpu_id}')
  else:
      device = torch.device('cpu')
  
  #Si la biblioteca cuDNN está disponible se activan las optimizaciones 
  if torch.backends.cudnn.is_available():
    #print('* Activando CUDNN')
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchhmark=True


  # experimentos deterministas o aleatorios
  #Si DET posee valor 1 el experimento será determinista
  if(semilla_fija == 1):
    SEED = SEMILLA + exp
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
  sizex=SIZEX; sizey=SIZEX 

  # 2. Load datos
  
  # Seleccionamos los conjuntos de entrenamiento, validación y test
  (train, val, test, nclases, nclases_no_vacias) = select_training_samples_seg(truth, center, H, V, sizex, sizey, SAMPLES)

  #Creamos el dataset de entrenamiento y el dataset de testeo en base a los conjuntos de entrenamiento y de testeo
  dataset_train=HyperDataset(datos,truth,train,H,V,sizex,sizey, is_train=True)
  dataset_test=HyperDataset(datos,truth,test,H,V,sizex,sizey, is_train=False)


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
    # replacement=True es CLAVE: permite repetir muestras minoritarias para rellenar huecos
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

  test_loader = DataLoader(dataset_test, batch_size, shuffle=False, num_workers=num_workers_dl)

  # Si queremos validacion
  if(len(val)>0):
    #Creamos el dataset de validación con el conjunto de validación
    dataset_val=HyperDataset(datos,truth,val,H,V,sizex,sizey, is_train=False)
    #Creamos el dataloader que se usará durante la validación de la red neuronal
    val_loader=DataLoader(dataset_val,batch_size,shuffle=False)
 
  # 4. Hyper parameters específicos para el Vision Transformer
  base_learning_rate=1e-3 #Tasa de aprendizaje inicial
  weight_decay=0.05 #Penalización de los pesos grandes (regularización L2 para evitar sobreajuste)
  mask_ratio=0.75 #(Usado típicamente en MAE - Masked Autoencoders)
  warmup_epoch=200 #Número de pasos de "calentamiento" para el learning rate
  model_path='/tmp/vit_v3.pt' #Ruta donde se guardará el mejor modelo

  # OJO: no todos estos parametros se usan en este transformer
  #Bloque de configuración manual de parámetros según el tamaño de la red elegido (ViTsize)
  #depth: número de capas de bloques Transformer
  #heads: número de cabezales de atención (Multi-Head Attention)
  #patch_size: tamaño en el que se subdivide la imagen original (ej. 8x8 o 4x4)
  #hidden_size_dim/embed_dim: dimensión del vector latente que representa cada patch
  #mlp_dim: dimensión oculta en el bloque Feed Forward dentro del Transformer
  if(ViTsize==0):
    depth=6; heads=4; patch_size=8; hidden_size_dim=256; mlp_dim=1024;
    dropout=0.1; emb_dropout=0.1;
  elif(ViTsize==1):
    depth=8; heads=8; patch_size=8; hidden_size_dim=512; mlp_dim=2048;
    dropout=0.1; emb_dropout=0.1;
  elif(ViTsize==2):
    depth=12; heads=12; patch_size=8; mlp_dim=3072; hidden_size_dim=768; 
    dropout=0.2; emb_dropout=0.1;
  elif(ViTsize==3):
    depth=24; heads=16; patch_size=8; mlp_dim=4096; hidden_size_dim=1024; 
    dropout=0.1; emb_dropout=0.1;
  elif(ViTsize==4):
    depth=8; heads=8; patch_size=4; mlp_dim=512; hidden_size_dim=128;
    dropout=0.1; emb_dropout=0.1;

  # 5. Red: Instanciamos el modelo usando la librería TIMM (PyTorch Image Models)
  if(ViTtype==0): model=timm.models.vision_transformer.VisionTransformer(
    #Modelo Vision Transformer estándar
    img_size=sizex, num_classes=nclases, in_chans=B, #Ajustamos tamaño, clases y bandas (B)
    patch_size=patch_size, depth=depth, num_heads=heads,  
    embed_dim=256, mlp_ratio=4., qkv_bias=False, drop_rate=0.,
    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm).to(device)
  
  else: 
    #Modelo Swin Transformer (utiliza atención mediante ventanas desplazadas - shifted windows)
    model=timm.models.SwinTransformer(
    img_size=sizex, num_classes=nclases, in_chans=B,
    patch_size=patch_size, window_size=16).to(device)


  # Inicializamos FMix
  fmix_util = FMix(size=(sizex, sizey), alpha=fmix_alpha, decay_power=fmix_decay, max_soft=fmix_soft)

  # 6. Loss, optimizer, and scheduler
  #Definimos la función de pérdida. Usamos entropía cruzada.
  loss_fn=torch.nn.CrossEntropyLoss()
  
  #Definimos una función anónima (lambda) para calcular el "accuracy" (acierto) sobre la marcha
  acc_fn=lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

  #Especificamos el modelo de optimización (AdamW: Adam con Weight Decay mejorado)
  optim=torch.optim.AdamW(model.parameters(), lr=base_learning_rate * batch_size / 256, betas=(0.9, 0.95), weight_decay=weight_decay)
  lr_func=lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch /EPOCHS * math.pi) + 1))

  #Scheduler que ajusta el learning rate: inicia con un calentamiento lineal (warmup) y luego decae siguiendo una curva del coseno
  lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

  #------------------------------------------------
  # TRAIN
  #------------------------------------------------

  best_val_acc=0
  step_count=0
  #Borramos los gradientes que pudieran estar acumulados antes de iniciar
  optim.zero_grad()

  tiempo_inicial_entrenamiento = time.perf_counter()
  
  #Bucle de entrenamiento asociado a las épocas
  for e in range(EPOCHS):
    #Activamos el modo entrenamiento del modelo
    model.train()
    losses=[]
    acces=[]
    #Recorremos los patches (inputs) junto a sus etiquetas (labels) del conjunto de entrenamiento
    for i,(img,label) in enumerate(train_loader):
      step_count += 1
      
      #Cargamos los datos y sus etiquetas en la GPU/CPU
      img=img.to(device)
      label=label.to(device)

      # APLICAMOS CUTMIX / FMIX CON PROBABILIDADES
      if(random.random() < probabilidad):
        if random.random() < probabilidad2:
          # --- INTEGRACIÓN FMIX ---
          inputs_mixed = fmix_util(img) 
          lam = fmix_util.lam         
          indices = fmix_util.index   
          logits = model(inputs_mixed)
          loss = lam * loss_fn(logits, label) + (1 - lam) * loss_fn(logits, label[indices])
          acc = acc_fn(logits, label) 
        else:
          # --- INTEGRACIÓN CUTMIX ---
          inputs_mixed, target_a, target_b, lam = aplicar_cutmix(img, label, alpha=cutmix_alpha)
          logits = model(inputs_mixed)
          loss = lam * loss_fn(logits, target_a) + (1 - lam) * loss_fn(logits, target_b)
          acc = acc_fn(logits, target_a) 
      else:
        # Sin aumentado
        logits=model(img)
        loss=loss_fn(logits, label)
        acc=acc_fn(logits, label)
      
      #Backward pass: Realizamos la retropropagación, calculando el gradiente
      loss.backward()
      #Actualizamos los pesos y sesgos de la red con el optimizador AdamW
      optim.step()

      #Borramos los gradientes para el siguiente batch
      optim.zero_grad()

      #Guardamos los errores y precisiones del batch actual
      losses.append(loss.item())
      acces.append(acc.item())

    # Actualizamos el learning rate al final de la época usando el scheduler del coseno
    lr_scheduler.step()

    
    # Calculamos la media de pérdida y precisión del entrenamiento en toda la época
    avg_train_loss=sum(losses) / len(losses)
    avg_train_acc=sum(acces) / len(acces)

    if(e==EPOCHS-1):
      tiempo_final_entrenamiento = time.perf_counter()
      tiempo_total_entrenamiento = tiempo_final_entrenamiento - tiempo_inicial_entrenamiento
      tiempo_epoca_entrenamiento = tiempo_total_entrenamiento / EPOCHS

    # si tenemos validacion usamos estas muestras
    if(len(val)>0):
      
      #Ponemos el modelo en modo evaluación (evitando que se actualicen las estadísticas de BatchNorm/Dropout)
      model.eval()
      
      # INICIALIZAMOS CONTADORES PARA EL AA
      val_class_correct = [0] * (nclases + 1)
      val_class_total = [0] * (nclases + 1)

      #Desactivamos el cálculo de gradientes para ahorrar memoria y CPU/GPU
      with torch.no_grad():
        losses=[]

        #Recorremos batches de patches de validación
        for i,(img,label) in enumerate(val_loader):
          img=img.to(device)
          label=label.to(device)
          #Realizamos las predicciones
          logits=model(img)

          #Calculamos el error y el accuracy
          loss=loss_fn(logits, label)

          losses.append(loss.item())

          # SACAMOS LAS PREDICCIONES
          (_, predicted) = torch.max(logits, 1)

          # LLENAMOS LOS CONTADORES POR CLASE
          for j in range(len(label)):
            real_class = label[j].item() + 1 # +1 porque restamos 1 en el Dataset
            val_class_total[real_class] += 1
            if predicted[j] == label[j]:
              val_class_correct[real_class] += 1
          
          avg_val_loss = sum(losses) / len(losses)

          # CALCULAMOS EL AA DE VALIDACIÓN
          val_accuracies = []
          for c in range(1, nclases + 1):
            if val_class_total[c] > 0:
              val_accuracies.append(100 * val_class_correct[c] / val_class_total[c])
          
          current_val_aa = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0

          if(len(val)>0 and current_val_aa > best_val_acc):
            best_val_acc = current_val_aa     
            torch.save(model, model_path)

  #Si no está activado el flag de testeo la función devolvemos el average accuracy obtenido
  if(TEST==0): 
    if len(val) > 0: return current_val_aa
    else: return avg_train_acc

  # 8. Test the model
  #print('* Test ViT, exp.%d'%(exp))
  
  #Creamos el mapa de salida de clasificación de los píxeles con todo 0
  output=np.zeros(H*V,dtype=np.uint8) # mapa de salida de pixels
  # OJO, sale mejor usar el modelo de la ultima-iteracion
  # model=torch.load(model_path, map_location='cpu').to(device)

  #Ponemos el modelo en modo evaluación
  model.eval()

  #Evitamos que se almacene el grafo asociado a los gradientes
  with torch.no_grad():
    correct=0; total=0;

    #Realizamos la clasificación sobre el conjunto de test
    for(inputs,labels) in test_loader:
      #Cargamos los patches y sus etiquetas
      inputs=inputs.to(device)
      labels=labels.to(device)

      #Realizamos la clasificación de los patches
      outputs=model(inputs)

      #Obtenemos la clasificación final asociada a cada patch (la clase con mayor valor)
      (_,predicted)=torch.max(outputs.data,1)
      
      #Cargamos las predicciones en la RAM
      predicted_cpu=predicted.cpu()

      #Recorremos las predicciones realizadas para cada uno de los patches
      for i in range(len(predicted_cpu)):
        #La red devuelve las clases empezando en 0, por tanto sumamos 1
        #Se guarda la predicción de cada patch en la posición asociada al centro de segmento
        output[test[total+i]]=np.uint8(predicted_cpu[i]+1)
      
      #Aumentamos la variable total con tantas posiciones como patches se hayan procesado
      total+=labels.size(0)
      
      #Cada vez que se han clasificado 2000 patches se imprime por pantalla el progreso
      #if(total%2000==0): print('  Test: %6d/%d'%(total,len(dataset_test)))

  #Tras lo anterior tenemos el mapa con únicamente la clasificación de los centros. Hay que propagarlo.
  #Recorremos todos los píxeles y miramos qué clase fue asignada al píxel central del segmento al que pertenece
  for i in range(H*V): output[i]=output[center[seg[i]]]
  # Eliminamos los centros usados en el entrenamiento y validación de la red poniéndolos a 0
  for i in train: output[i]=0
  for i in val: output[i]=0
  
  # 9. precisiones por segmentos (excluyendo los usados en el entrenamiento y validación)
  correct=0; total=0
  
  #Recorremos todos los centros de segmentos
  for i in range(len(center)):
    #Si en el output el centro tiene un valor 0 no lo contabilizamos (usado en train/val)
    if(output[center[i]]==0): continue
    
    #Aumentamos el contador de segmentos totales a evaluar
    total+=1
    
    #Si coinciden predicción y verdad, aumentamos los aciertos
    if(output[center[i]]==truth[center[i]]): correct=correct+1
  
  # Calculamos el accuracy a nivel de segmento
  acc=100*correct/total;
  #print('* Accuracy (segments): %.02f'%(acc))

  # 10. precisiones a nivel de pixel
  #Creamos los contadores y listas necesarias para calcular el overall accuracy y el average accuracy
  correct=0; total=0; AA=0; OA=0
  class_correct=[0]*(nclases+1)
  class_total=[0]*(nclases+1)
  class_aa=[0]*(nclases+1)

  #Recorremos los píxeles que fueron clasificados por la red
  for i in range(len(output)):
    #Si el píxel fue empleado en entrenamiento/validación o no está clasificado en las etiquetas originales, se ignora
    if(output[i]==0 or truth[i]==0): continue
    #Aumentamos contadores totales
    total+=1; class_total[truth[i]]+=1

    #Si la predicción coincide con la etiqueta real, aumentamos contadores de éxito
    if(output[i]==truth[i]):
      correct+=1
      class_correct[truth[i]]+=1

  #Ahora pasamos a calcular el accuracy de cada clase
  for i in range(1,nclases+1):
    #Si existen píxeles de la clase actual, calculamos porcentaje de accuracy
    if(class_total[i]!=0): class_aa[i]=100*class_correct[i]/class_total[i]
    else: class_aa[i]=0

    #Sumamos el accuracy para luego hacer la media (AA)
    AA+=class_aa[i]

  #Calculamos el Overall Accuracy y el Average Accuracy
  OA=100*correct/total; AA=AA/nclases_no_vacias 
  #print('* Accuracy (pixels) exp.%d:'%(exp))
  
  
  #Finalizamos devolviendo el Overall Accuracy, Average Accuracy y el accuracy individual de las clases
  return (OA, AA, class_aa, class_total, tiempo_total_entrenamiento, tiempo_epoca_entrenamiento)



def run_combination(params_with_data):
    torch.set_num_threads(1)
    gpu_id, params, data_bundle = params_with_data
    a_fmix, d, s, a_cmix, e, b, p, p2, samp= params
    
    val_acc_list = []
    print(f"[GPU: {gpu_id}] Evaluando ViT: F_Alpha={a_fmix}, Decay={d}, Soft={s}, C_Alpha={a_cmix}, Epochs={e}, Batch={b}, Prob={p}, Prob2={p2}, Usar_sampler={samp}")
    sys.stdout.flush()

    for exp in range(1):
        res = main(exp, a_fmix, d, s, a_cmix, data_bundle, 0, e, b, p, p2, samp, 1, gpu_id)
        v_acc = res[0] if isinstance(res, tuple) else res
        val_acc_list.append(v_acc)

    return {'fmix_alpha': a_fmix, 'decay': d, 'soft': s, 'cutmix_alpha': a_cmix, 'epochs':e,'batch':b ,'prob':p, 'prob2':p2, 'sampler':samp, 'mean_val_aa': np.mean(val_acc_list)}


def run_final_eval(args):
    torch.set_num_threads(1)
    gpu_id, exp_idx, fmix_alpha, decay, soft, cutmix_alpha, epochs, batch, prob, prob2, samp, data_bundle = args
    oa, aa, class_aa, class_total, tiempo_total_entrenamiento, tiempo_epoca = main(exp_idx, fmix_alpha, decay, soft, cutmix_alpha, data_bundle, 1, epochs, batch, prob, prob2, samp, DET, gpu_id)
    return oa, aa, class_aa, class_total, tiempo_total_entrenamiento, tiempo_epoca


#Si se lanza el fichero directamente se entra en el entrenamiento y validación
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"GPUs detectadas: {num_gpus}")
    
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    directorio_datos = os.path.join(directorio_actual, '..', 'datosEntrada')

    #Si no se ha indicado un número asociado a un dataset se ejecuta la prueba asociada al dataset del río Oitaven
    if len(sys.argv) < 3:
        ficheroLeido = "oitaven"
        try:
            usar_sampler = int(sys.argv[1])
        except ValueError:
            print("Error: El argumento debe ser un número entero.")
            sys.exit(1)

        print("********************Ejecutando prueba sobre el dataset " + ficheroLeido + " ******************************")
        #Dataset: dataset original, contiene la información obtenida por el dron (cada píxel tiene un cierto número de bandas con datos en cada una)
        DATASET = os.path.join(directorio_datos, 'oitaven', 'oitaven_river.raw')
        #GT: Etiquetas de cada segmento, son las etiquetas reales correspondientes a cada segmento, los segmentos son de 32 x 32 píxeles centrados en un centro.
        GT = os.path.join(directorio_datos, 'oitaven', 'oitaven_river.pgm')
        #SEG: segmentación, cada píxel tiene el ID del segmento al que pertenece
        SEG = os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp.raw')
        #CENTER: centros de los segmentos, contiene los índices de cada píxel correspondiente al centro de cada segmento.
        CENTER = os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp_centers.raw')
    else:
        try:
            opcion = int(sys.argv[1])
            usar_sampler = int(sys.argv[2])
        except ValueError:
            print("Error: El argumento debe ser un número entero.")
            sys.exit(1)
        
        match opcion:
            case 1:
                ficheroLeido="das_mestas"
                DATASET= os.path.join(directorio_datos, 'das_mestas', 'das_mestas_river.raw')
                GT= os.path.join(directorio_datos, 'das_mestas', 'das_mestas_river.pgm')
                SEG= os.path.join(directorio_datos, 'das_mestas', 'seg_mestas_wp.raw')
                CENTER= os.path.join(directorio_datos, 'das_mestas', 'seg_mestas_wp_centers.raw')
            case 2:
                ficheroLeido="eiras_dam"
                DATASET= os.path.join(directorio_datos, 'eiras_dam', 'eiras_dam.raw')
                GT= os.path.join(directorio_datos, 'eiras_dam', 'eiras_dam.pgm')
                SEG= os.path.join(directorio_datos, 'eiras_dam', 'seg_eiras_wp.raw')
                CENTER= os.path.join(directorio_datos, 'eiras_dam', 'seg_eiras_wp_centers.raw')
            case 3:
                ficheroLeido="ermidas_creek"
                DATASET= os.path.join(directorio_datos, 'ermidas_creek', 'ermidas_creek.raw')
                GT= os.path.join(directorio_datos, 'ermidas_creek', 'ermidas_creek.pgm')
                SEG= os.path.join(directorio_datos, 'ermidas_creek', 'seg_ermidas_wp.raw')
                CENTER= os.path.join(directorio_datos, 'ermidas_creek', 'seg_ermidas_wp_centers.raw')
            case 4:
                ficheroLeido="ferreiras_river"
                DATASET= os.path.join(directorio_datos, 'ferreiras_river', 'ferreiras_river.raw')
                GT= os.path.join(directorio_datos, 'ferreiras_river', 'ferreiras_river.pgm')
                SEG= os.path.join(directorio_datos, 'ferreiras_river', 'seg_ferreiras_wp.raw')
                CENTER= os.path.join(directorio_datos, 'ferreiras_river', 'seg_ferreiras_wp_centers.raw')
            case 5:
                ficheroLeido="mera_river"
                DATASET= os.path.join(directorio_datos, 'mera_river', 'mera_river.raw')
                GT= os.path.join(directorio_datos, 'mera_river', 'mera_river.pgm')
                SEG= os.path.join(directorio_datos, 'mera_river', 'seg_mera_wp.raw')
                CENTER= os.path.join(directorio_datos, 'mera_river', 'seg_mera_wp_centers.raw')
            case 6:
                ficheroLeido="ulla"
                DATASET = os.path.join(directorio_datos, 'ulla', 'ulla_river.raw')
                GT= os.path.join(directorio_datos, 'ulla', 'ulla_river.pgm')
                SEG= os.path.join(directorio_datos, 'ulla', 'seg_ulla_wp.raw')
                CENTER= os.path.join(directorio_datos, 'ulla', 'seg_ulla_wp_centers.raw')
            case 7:
                ficheroLeido="xesta"
                DATASET= os.path.join(directorio_datos, 'xesta', 'xesta_basin.raw')
                GT= os.path.join(directorio_datos, 'xesta', 'xesta_basin.pgm')
                SEG= os.path.join(directorio_datos, 'xesta', 'seg_xesta_wp.raw')
                CENTER= os.path.join(directorio_datos, 'xesta', 'seg_xesta_wp_centers.raw')
            case _:
                ficheroLeido="oitaven"
                DATASET= os.path.join(directorio_datos, 'oitaven', 'oitaven_river.raw')
                GT= os.path.join(directorio_datos, 'oitaven', 'oitaven_river.pgm')
                SEG= os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp.raw')
                CENTER= os.path.join(directorio_datos, 'oitaven', 'seg_oitaven_wp_centers.raw')
        print("********************Ejecutando prueba sobre el dataset " + ficheroLeido + " ******************************")
    
    # 1. CARGA LOS DATOS UNA SOLA VEZ AQUÍ
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

    # Creamos el bundle
    data_bundle = {
        'datos': datos_tensor,
        'H': H, 'V': V, 'B': B,
        'truth': truth, 'H1': H1, 'V1': V1,
        'seg': seg, 'H2': H2, 'V2': V2,
        'center': center, 'H3': H3, 'V3': V3,
        'nseg': nseg
    }

    if(usar_sampler==1):
      archivoParametros = "hiperParametros_ViT_FMIX_CUTMIX_Con_Aumentado.json"
    else:
      archivoParametros = "hiperParametros_ViT_FMIX_CUTMIX.json"

    mejor_config = None

    if os.path.exists(archivoParametros):
      print(f"--- Cargando hiperparámetros óptimos desde {archivoParametros} ---")
      with open(archivoParametros, 'r') as f:
        mejor_config = json.load(f)

    if mejor_config is None:
      if ficheroLeido != "oitaven":
        print("ERROR: No hay parámetros optimizados almacenados para ViT")
        sys.exit(1)
      else:
        # ---> GRID SEARCH DE VIT FMIX+CUTMIX
        fmix_alphas = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
        decays =  [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5]
        softs  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        cutmix_alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        probs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        probs2 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        PRUEBAS=200
        
        combinaciones_totales = list(itertools.product(
          fmix_alphas, decays, softs, cutmix_alphas, [100], [256], probs, probs2, [usar_sampler]
        ))

        # Seleccionamos N_TRIALS al azar si exceden el límite
        if len(combinaciones_totales) > PRUEBAS:
          combinaciones = random.sample(combinaciones_totales, PRUEBAS)
        else:
          combinaciones = combinaciones_totales

        tareas = [(i % num_gpus, comb, data_bundle) for i, comb in enumerate(combinaciones)]
        
        print(f"--- Iniciando Grid Search Paralelo ViT ({len(combinaciones)} combinaciones) ---")

        with ProcessPoolExecutor(max_workers=4) as executor:
            resultados_finales = list(executor.map(run_combination, tareas))
            executor.shutdown(wait=True)

        time.sleep(0.5)

        # Seleccionar el mejor
        resultados_finales.sort(key=lambda x: x['mean_val_aa'], reverse=True)
        mejor_config = resultados_finales[0]
        
        # ---> ¡NUEVO! <--- Guardar JSON
        with open(archivoParametros,'w') as f:
          json.dump(mejor_config,f)

    print(f"\n--- Ejecutando evaluación final paralela ViT ({EXP} experimentos) ---")

    # Especificamos los parámetros asociados al experimento
    tareas_finales = [
        (i % num_gpus, i, mejor_config['fmix_alpha'], mejor_config['decay'], mejor_config['soft'], mejor_config['cutmix_alpha'], mejor_config['epochs'], mejor_config['batch'], mejor_config['prob'], mejor_config['prob2'], mejor_config['sampler'], data_bundle) 
        for i in range(EXP)
    ]
    print("Ejecutando test de ViT...")
    
    #Ejecutamos el test con 2 procesos
    with ProcessPoolExecutor(max_workers=4) as executor:
        resultados_test = list(executor.map(run_final_eval, tareas_finales))
        executor.shutdown(wait=True)
    
    time.sleep(0.5)

    # 4. EXTRACCIÓN Y CÁLCULO DE ESTADÍSTICAS
    final_oa_list = [res[0] for res in resultados_test]
    final_aa_list = [res[1] for res in resultados_test]
    class_aa_matrix = np.array([res[2] for res in resultados_test])
    class_total_matrix = np.array([res[3] for res in resultados_test])

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

    # 5. IMPRESIÓN DE RESULTADOS FINALES
    print("\n" + "="*60)
    print("RESULTADOS FINALES PROMEDIADOS VIT+FMIX+CUTMIX SOBRE: " + ficheroLeido)
    print("="*60)
    print(f"Mejor Configuración ViT:")
    print(f" FMIX_Alpha={mejor_config['fmix_alpha']}, Decay={mejor_config['decay']}, Soft={mejor_config['soft']}")
    print(f" CUTMIX_Alpha={mejor_config['cutmix_alpha']}")
    print(f" Epoch={mejor_config['epochs']}, Batch={mejor_config['batch']}, Prob_Principal={mejor_config['prob']}, Prob_Metodo2={mejor_config['prob2']}, Sampler={mejor_config['sampler']}")
    print("-" * 60)
    
    print(f"ACCURACY POR CLASE:")
    for j in range(1, len(m_class)): 
        if m_total[j] > 0: 
            print(f"  Clase {j:02d}: {m_class[j]:.2f}% ± {s_class[j]:.2f}%")

    print("-" * 60)
    print(f"OA Final: {m_oa:.2f}% ± {s_oa:.2f}%")
    print(f"AA Final: {m_aa:.2f}% ± {s_aa:.2f}%")
    print("-" * 60)

    print(f"Tiempo entrenamiento total medio con hiperparámetros óptimos: {m_t_total:.2f} s")
    print(f"Tiempo medio por época con hiperparámetros óptimos: {m_t_epoch:.4f} s")
    print("="*60)