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
from torch.utils.data import Dataset,DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms
import sklearn.utils.class_weight as class_weight
import torch.nn.functional as F
from einops import repeat
import timm

EXP=5      # numero de experimentos (NÚMERO DE VECES QUE SE REPITE EL PROCESO DE ENTRENAMIENTO Y PRUEBA), los resultados serán el promedio de cada resultado
SAMPLES=[0.15,0.05] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05) (PORCENTAJE DE ENTRENAMIENTO (segmentos usados para entrenar), PORCENTAJE DE VALIDACIÓN (segmentos usados para validar))
EPOCHS=100 # EPOCHS de entrenamiente del clasificador (defecto=100)  **[NO CAMBIES ESTO]**
BATCH=256  # batch-size, defecto=100 
SIZEX=32   # tamano del patch (defecto=32)  **[NO CAMBIES ESTO]**
DET=0      # experimentos: 0-aleatorios, 1-deterministas (defecto=0)
AUM=1      # aumentado: 0-sin_aumentado, 1-con_aumentado (defecto=1)
ViTsize=4  # 0-micro, 1-mini, 2-base, 3-large, 4-pruebas (Tamaño y complejidad del modelo Transformer a utilizar)
ViTtype=0  # 0-vit, 1-swin (Tipo de arquitectura: 0 para Vision Transformer clásico, 1 para Swin Transformer)
TEST=1     # 0-validacion, 1-test
	   
DATASET='/mnt/media/images/oitaven_river.raw'
GT='/mnt/media/images/oitaven_river.pgm'
SEG='/mnt/media/images/seg_oitaven_wp.raw'
CENTER='/mnt/media/images/seg_oitaven_wp_centers.raw'

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
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  #Se imprime información sobre el dataset leído
  print('* Read dataset:',fichero)
  print('  B:',B,'H:',H,'V:',V)
  print('  Read:',len(datos))
  # esta red no necesita realmente normalizar
  #Aplicamos escalado Min-Max con rango [0,1]
  datos=preprocessing.minmax_scale(datos)
  print('  min:',datos.min(),'max:',datos.max())
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
  print('* Read segmentation:',fichero)
  print('  H:',H,'V:',V)
  print('  Read:',len(datos))
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
  #Se imprime información sobre el fichero de centros de segmentos leído
  print('* Read centers:',fichero)
  print('  H:',H,'V:',V,'nseg',nseg)
  print('  Read:',len(datos))
  #Devolvemos los datos sobre los centros de los segmentos junto a al anchura, la altura y el número de segmentos
  return(datos,H,V,nseg)

#Función que permite guardar el mapa de clasificación final en un nuevo fichero (versión byte a byte)
def save_raw(output,H,V,B,filename):
  #Tratamos de abrir el fichero en modo escritura binaria
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    f.write(struct.pack('i',B))
    f.write(struct.pack('i',H))
    f.write(struct.pack('i',V))
    output=output.reshape(H*V*B)
    for i in range(H*V*B):
      f.write(struct.pack('i',np.int(output[i])))
    f.close()
    print('* Saved file:',filename)

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
    print('* Read GT:',fichero)
    print('  H:',H,'V:',V,'depth:',depth)
    print('  Read:',len(raster))
    return(raster,H,V)


#Función que permite almacenar en un fichero PGM las predicciones realizadas por la red neuronal para cada píxel
def save_pgm(output,H,V,nclases,filename):
  #Abrimos el fichero en modo escritura binaria
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    # f.write(b'P5\n')
    #Se construye la cabecera del fichero PGM (nclases es el número de clases en total)
    cadena='P5\n'+str(H)+' '+str(V)+'\n'+str(nclases)+'\n'
    #Se elmacena la cabecera en bytes usando la codificación utf-8
    f.write(bytes(cadena,'utf-8'))
    #Se almacenan la clasificación de los píxeles
    f.write(output)
    f.close()
    print('* Saved file:',filename)


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




# Esta parte tarda mucho, mejor la preprocesamos en C
#Esta función es la encargada de obtener el punto central de cada segmento de la imagen (es muy lento y no se usa)
def seg_center(seg,H,V):
  print('* Segment centers (tarda mucho)')
  nseg=0
  #Se comprueba cuantos segmentos hay en total
  for i in range(H*V):
    if(seg[i]>nseg): nseg=seg[i]
  nseg=nseg+1
  print('  segments:',nseg)

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
  print('* Select training samples')
  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)

  #Recorremos el fichero con las etiquetas para saber cuántas clases hay en total
  for i in truth:
    if(i>nclases): nclases=i
  print('  nclasses:',nclases)
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
  print('  Class  # :   total | train |   val |    test')
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
    #Se imprime un resumen sobre la repartición de los datos en los distintos conjuntos en función de la clase
    print('  Class',f'{i+1:2d}',':',f'{len(lista[i]):7d}','|',f'{tot0:5d}','|',
      f'{tot1:5d}','|',f'{len(lista[i])-tot0-tot1:7d}')
  return(train,val,test,nclases,nclases_no_vacias)

#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------

# cogemos muestras sin ground-truth (dadas por el indice samples)
#Clase asociada al dataset sin el ground truth (etiquetas), se usa para realizar la inferencia final sobre la imagen entera (aunque tengan etiqueta o no de ground-truth)
class HyperAllDataset(Dataset):
  def __init__(self,datos,samples,H,V,sizex,sizey):
    #Se guarda la imagen (datos) y los índices de los centros (samples) de los segmentos
    self.datos=datos; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    #Herramienta de aumentado de datos, se realizan estas operaciones con probabilidad al azar
    self.transform=transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip()])
    
  #Función para devolver el número de instancias del dataset
  def __len__(self):
    return len(self.samples)

  #Función que se ejecuta cada vez que se pide un patch para analizar
  def __getitem__(self,idx):
    #Se recuperan los datos almacenados al construir la instancia
    datos=self.datos; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    #Se convierte el índice del centroide en coordenadas 2D (x e y)
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    #Se obtiene el patch alrededor de ese centroide
    patch=select_patch(datos,sizex,sizey,x,y)
    #Si el aumentado de datos está activado se aplican las transformaciones al azar
    if(AUM==1): patch=self.transform(patch)
    #Se devuelve el patch final
    return(patch)

#----------------

# cogemos muestras con ground-truth (dadas por el indice samples)
#Clase asociada al dataset con las etiquetas, igual que el anterior pero con las etiquetas del ground truth. Usada para el entrenamiento.
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey):
    #Se guarda la imagen (datos), las etiquetas asociadas a los píxeles y los índices de los centros (samples) de los segmentos
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    #Herramienta de aumentado de datos
    self.transform=transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip()])
    
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
    if(AUM==1): patch=self.transform(patch)
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



# calcula los promedios de precisiones
#Función que permite calcular los promedios y desviaciones de la precisión del modelo para cada clase a partir de los resultados de los diversos experimentos
#OA es el Overall Accuracy, es la lista con el acierto total de cada experimento (Total aciertos en todas las clases / Total muestras de todas las clases)
#AA es el Average Accuracy, es la lista con la media de aciertos de las clases de cada experimento (Sumatorio del porcentaje de acierto de cada clase / número de clases)
#aa es la matriz donde cada fila tiene el acierto individual de cada clase por cada experimento (en porcentaje)
def accuracy_mean_deviation(OA,AA,aa):
  #Número de experimentos realizados
  n=len(OA); nclases=len(aa[0])
  print('* Means and deviations (%d exp):'%(n))
  # medias
  #Pasamos a calcular la media de Overall Accuracy y del Average Accuracy
  OAm=0; AAm=0; aam=[0]*nclases;
  
  for i in range(n):
     #Sumamos el Overall Accuracy y el Average Accuracy de cada experimento
     OAm+=OA[i]; AAm+=AA[i]
     #Se suma el acierto en cada una de las clases
     for j in range(1,nclases): aam[j]+=aa[i][j]
  
  #Obtenemos el Overal Accuracy medio y el Average Accuracy medio
  OAm/=n; AAm/=n
  
  #Calculamos la media de aciertos para cada clase
  for j in range(1,nclases): aam[j]/=n
  # desviaciones, usamos la formula que divide entre (n-1)
  OAd=0; AAd=0; aad=[0]*nclases

  #Pasamos a calcular la desviación usando la fórmula de desviación estándar muestral
  for i in range(n):
     #Sumamos los cuadrados de las diferencias respecto a la media
     OAd+=(OA[i]-OAm)*(OA[i]-OAm); AAd+=(AA[i]-AAm)*(AA[i]-AAm)
     #Sumamos el cuadrado de las diferencias respecto a la media para cada clase
     for j in range(1,nclases): aad[j]+=(aa[i][j]-aam[j])*(aa[i][j]-aam[j])
  
  #Calculamos la desviación estándar del Overall Accuracy y del Average Accuracy
  OAd=math.sqrt(OAd/(n-1)); AAd=math.sqrt(AAd/(n-1))
  #Calculamos la desviación estándar por clase
  for j in range(1,nclases): aad[j]=math.sqrt(aad[j]/(n-1))
  for j in range(1,nclases): print('  Class %02d: %02.02f+%02.02f'%(j,aam[j],aad[j]))
  print('  OA=%02.02f+%02.02f, AA=%02.02f+%02.02f'%(OAm,OAd,AAm,AAd))

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

def main(exp):
  print('* ViT exp: '+str(exp))
  #Tomamos la primera referencia temporal antes de realizar el entrenamiento
  time_start=time.time()


  # 1. Device configuration
  #Comprobamos si el sistema tiene una gráfica compatible con CUDA disponible, si es así se pasa a usar la GPU para entrenar y ejecutar el modelo
  cuda=True if torch.cuda.is_available() else False
  print('* Cuda: '+str(cuda))
  device=torch.device('cuda' if cuda else 'cpu')
  
  #Si la biblioteca cuDNN está disponible se activan las optimizaciones 
  if torch.backends.cudnn.is_available():
    print('* Activando CUDNN')
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchhmark=True


  # experimentos deterministas o aleatorios
  #Si DET posee valor 1 el experimento será determinista
  if(DET==1):
    #Fijamos la semilla a 0
    SEED=0
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

  # 2. Load datos
  #Cargamos los datos en crudo (la imagen original)
  #En datos se guardan todos los píxeles y sus bandas
  (datos,H,V,B)=read_raw(DATASET)
  #Cargamos el Ground Truth, el cual tentrá un ancho H1 y un alto V1
  (truth,H1,V1)=read_pgm(GT)
  #Cargamos la segmentación de los píxeles, el cual tendrá un ancho H2 y un alto V2
  (seg,H2,V2)=read_seg(SEG)
  # necesitamos los datos en band-vector para hacer convoluciones
  #Reordenamos los datos para que puedan ser empleados en las redes neuronales de PyTorch, pasamos las bandas de la posición 2 a la posición 0
  datos=np.transpose(datos,(2,0,1))

  # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
  #Los patches serán de 32x32 píxeles
  sizex=SIZEX; sizey=SIZEX 

  # 3. Selection training,testing sets
  # (center,nseg)=seg_center(seg,H,V) # lento, mejor lo cargamos hecho
  #Cargamos las coordenadas de los centros de los segmentos
  (center,H3,V3,nseg)=read_seg_centers(CENTER)
  #Seleccionamos los conjuntos de entrenamiento, validación y test según lo especificado.
  (train,val,test,nclases,nclases_no_vacias)=select_training_samples_seg(truth,center,H,V,sizex,sizey,SAMPLES)

  #Creamos el dataset de entrenamiento y el dataset de testeo en base a los conjuntos de entrenamiento y de testeo
  dataset_train=HyperDataset(datos,truth,train,H,V,sizex,sizey)
  print('  - train dataset:',len(dataset_train))
  dataset_test=HyperDataset(datos,truth,test,H,V,sizex,sizey)
  print('  - test dataset:',len(dataset_test))


  # Dataloader
  #Indicamos el batch size (cantidad de patches que se van a procesar al mismo tiempo tanto para entrenar como para validar)
  batch_size=BATCH # defecto 100
  #Creamos el dataloader que se usará durante el entrenamiento
  #Con shuffle=True mezclamos los patches que se usan para entrenar, evitando que el modelo aprenda el orden
  train_loader=DataLoader(dataset_train,batch_size,shuffle=True)
  #Creamos el dataloader que se usará durante el testeo de la red neuronal, shuffle=False
  test_loader=DataLoader(dataset_test,batch_size,shuffle=False)
  # Si queremos validacion
  if(len(val)>0):
    #Creamos el dataset de validación con el conjunto de validación
    dataset_val=HyperDataset(datos,truth,val,H,V,sizex,sizey)
    print('  - val dataset:',len(dataset_val))
    #Creamos el dataloader que se usará durante la validación de la red neuronal
    val_loader=DataLoader(dataset_val,batch_size,shuffle=True)
 
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

  # 6. Loss, optimizer, and scheduler
  #Definimos la función de pérdida. Usamos entropía cruzada.
  loss_fn=torch.nn.CrossEntropyLoss()
  
  #Definimos una función anónima (lambda) para calcular el "accuracy" (acierto) sobre la marcha
  acc_fn=lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

  #Especificamos el modelo de optimización (AdamW: Adam con Weight Decay mejorado)
  optim=torch.optim.AdamW(model.parameters(), lr=base_learning_rate * batch_size / 256, betas=(0.9, 0.95), weight_decay=weight_decay)
  lr_func=lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch /EPOCHS * math.pi) + 1))

  #Scheduler que ajusta el learning rate: inicia con un calentamiento lineal (warmup) y luego decae siguiendo una curva del coseno
  lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

  #------------------------------------------------
  # TRAIN
  #------------------------------------------------

  best_val_acc=0
  step_count=0
  #Borramos los gradientes que pudieran estar acumulados antes de iniciar
  optim.zero_grad()

  
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

      #Forward pass: La red procesa los patches y devuelve sus predicciones
      logits=model(img)

      #Calculamos el error (pérdida) de las predicciones
      loss=loss_fn(logits, label)

      #Calculamos la precisión (accuracy)
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

    # si tenemos validacion usamos estas muestras
    if(len(val)>0):
      
      #Ponemos el modelo en modo evaluación (evitando que se actualicen las estadísticas de BatchNorm/Dropout)
      model.eval()
      
      #Desactivamos el cálculo de gradientes para ahorrar memoria y CPU/GPU
      with torch.no_grad():
        losses=[]
        acces=[]

        #Recorremos batches de patches de validación
        for i,(img,label) in enumerate(val_loader):
          img=img.to(device)
          label=label.to(device)
          #Realizamos las predicciones
          logits=model(img)

          #Calculamos el error y el accuracy
          loss=loss_fn(logits, label)
          acc=acc_fn(logits, label)

          #Almacenamos el error y el accuracy del batch actual
          losses.append(loss.item())
          acces.append(acc.item())
        
        #Calculamos el error medio y el accuracy medio obtenidos sobre el conjunto de validación
        avg_val_loss=sum(losses) / len(losses)
        avg_val_acc=sum(acces) / len(acces)
    
    #Imprimimos los resultados de la época
    if(len(val)>0): print(f'* Epoch {e}, Train loss: %02.04f, Acc: %02.04f, Val. loss: %02.04f, Acc: %02.04f'
       %(avg_train_loss,avg_train_acc,avg_val_loss,avg_val_acc))
    else: print(f'* Epoch {e}, Training loss: %02.04f, Acc: %02.04f'%(avg_train_loss,avg_train_acc))

    #Si la precisión de validación actual es la mejor hasta ahora, guardamos los pesos del modelo
    if(len(val)>0 and avg_val_acc > best_val_acc):
      best_val_acc=avg_val_acc
      print(f'  (saving best validation model with acc %02.04f at %d epoch)'%(best_val_acc,e))       
      torch.save(model,model_path)

  #Si no está activado el flag de testeo la función devuelve la media del accuracy obtenido
  if(TEST==0): return(sum(acces)/len(acces))

  # 8. Test the model
  print('* Test ViT, exp.%d'%(exp))
  
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
      if(total%2000==0): print('  Test: %6d/%d'%(total,len(dataset_test)))

  #Tras lo anterior tenemos el mapa con únicamente la clasificación de los centros. Hay que propagarlo.
  print('* Generating classif.map')
  
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
  print('* Accuracy (segments): %.02f'%(acc))

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

    #Suma el accuracy para luego hacer la media (AA)
    AA+=class_aa[i]

  #Calculamos el Overall Accuracy y el Average Accuracy
  OA=100*correct/total; AA=AA/nclases_no_vacias 
  print('* Accuracy (pixels) exp.%d:'%(exp))
  
  #Imprimimos el accuracy asociado a cada clase
  for i in range(1,nclases+1): print('  Class %02d: %02.02f'%(i,class_aa[i]))

  #Imprimimos el Overall Accuracy y el Average Accuracy a nivel de píxeles
  print('* Accuracy (pixels) exp.%d, OA=%02.02f, AA=%02.02f'%(exp,OA,AA))
  print('  total:',total,'correct:',correct)

  # guardamos la salida de la clasificación
  save_pgm(output,H,V,nclases,'/home/amo/output_vit-'+str(exp)+'.pgm')
  # Guardamos el modelo entrenado
  torch.save(model.state_dict(),'/tmp/model_vit-'+str(exp)+'.ckpt')

  #Tomamos la referencia de tiempo final
  time_end=time.time()
  print('* Execution time: %.0f s'%(time_end-time_start))
  print('  lr_base:',base_learning_rate,' imprime aqui otros parametros relevantes')

  #Finalizamos devolviendo el Overall Accuracy, Average Accuracy y el accuracy individual de las clases
  return(OA,AA,class_aa)

#Si se lanza el fichero directamente se entra en el entrenamiento y validación
if __name__=='__main__':
  #Si el flag TEST está a 0 no se ejecuta el modelo sobre el conjunto de test
  if(TEST==0): # validacion (se podria hacer una validacion cruzada de varias vias)
    acces=0
    # Se ejecuta la función main tantas veces como indique EXP
    for exp in range(EXP): acces=acces+main(exp)
    print('* ViT TIMM SEG EXP:',EXP,'SAMPLES:',SAMPLES,'EPOCHS:',EPOCHS,'BATCH:',BATCH,'SIZEX:',SIZEX,'AUM:',AUM)
    print('  VAL: %02.02f'%(100*acces/EXP))
    
  #Si el flag test está a 1 se realiza el test del modelo sobre el conjunto
  else: # test de 5 experimentos
    # Listas para guardar resultados de los diferentes experimentos
    OA=[0]*EXP; AA=[0]*EXP; aa=[0]*EXP 
    
    # Se ejecuta la función main y se almacenan los valores devueltos
    for exp in range(EXP): (OA[exp],AA[exp],aa[exp])=main(exp)
    
    # Si se ha realizado más de un experimento se calcula la media y la desviación típica
    if(EXP>1): accuracy_mean_deviation(OA,AA,aa) 
    print('* ViT TIMM SEG EXP:',EXP,'SAMPLES:',SAMPLES,'EPOCHS:',EPOCHS,'BATCH:',BATCH,'SIZEX:',SIZEX,'AUM:',AUM)
