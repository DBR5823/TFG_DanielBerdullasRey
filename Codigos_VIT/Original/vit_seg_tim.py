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

EXP=5      # numero de experimentos
SAMPLES=[0.15,0.05] # [entrenamiento,validacion]: muestras/clase (200,50) o porcentaje (0.15,0.05) 
EPOCHS=100 # EPOCHS de entrenamiente del clasificador (defecto=100)  **[NO CAMBIES ESTO]**
BATCH=100  # batch-size, defecto=100 
SIZEX=32   # tamano del patch (defecto=32)  **[NO CAMBIES ESTO]**
DET=0      # experimentos: 0-aleatorios, 1-deterministas (defecto=0)
AUM=1      # aumentado: 0-sin_aumentado, 1-con_aumentado (defecto=1)
ViTsize=4  # 0-micro, 1-mini, 2-base, 3-large, 4-pruebas
ViTtype=0  # 0-vit, 1-swin
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

def read_raw(fichero):
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  print('* Read dataset:',fichero)
  print('  B:',B,'H:',H,'V:',V)
  print('  Read:',len(datos))
  # esta red no necesita realmente normalizar
  datos=preprocessing.minmax_scale(datos)
  print('  min:',datos.min(),'max:',datos.max())
  datos=datos.reshape(V,H,B)
  datos=torch.FloatTensor(datos)
  return(datos,H,V,B)

def read_seg(fichero):
  (H,V)=np.fromfile(fichero,count=2,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=2*4,dtype=np.uint32)
  print('* Read segmentation:',fichero)
  print('  H:',H,'V:',V)
  print('  Read:',len(datos))
  return(datos,H,V)

def read_seg_centers(fichero):
  (H,V,nseg)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=3*4,dtype=np.uint32)
  print('* Read centers:',fichero)
  print('  H:',H,'V:',V,'nseg',nseg)
  print('  Read:',len(datos))
  return(datos,H,V,nseg)

def save_raw(output,H,V,B,filename):
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

def read_pgm(fichero):
  try:
    pgmf=open(fichero,"rb")
  except IOError:
    print('No puedo abrir ',fichero)
  else:
    assert pgmf.readline().decode()=='P5\n'
    line=pgmf.readline().decode()
    while(line[0]=='#'):
      line=pgmf.readline().decode()
    (H,V)=line.split()
    H=int(H); V=int(V)
    depth=int(pgmf.readline().decode())
    assert depth<=255
    raster=[]
    for i in range(H*V):
      raster.append(ord(pgmf.read(1)))
    print('* Read GT:',fichero)
    print('  H:',H,'V:',V,'depth:',depth)
    print('  Read:',len(raster))
    return(raster,H,V)

def save_pgm(output,H,V,nclases,filename):
  try:
    f=open(filename,"wb")
  except IOError:
    print('No puedo abrir ',filename)
    exit(0)
  else:
    # f.write(b'P5\n')
    cadena='P5\n'+str(H)+' '+str(V)+'\n'+str(nclases)+'\n'
    f.write(bytes(cadena,'utf-8'))
    f.write(output)
    f.close()
    print('* Saved file:',filename)

def select_patch(datos,sizex,sizey,x,y):
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2));     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2));
  patch=datos[:,y1:y2,x1:x2]
  return(patch)

# Esta parte tarda mucho, mejor la preprocesamos en C
def seg_center(seg,H,V):
  print('* Segment centers (tarda mucho)')
  nseg=0
  for i in range(H*V):
    if(seg[i]>nseg): nseg=seg[i]
  nseg=nseg+1
  print('  segments:',nseg)
  xmin=[H*V]*nseg; xmax=[0]*nseg; 
  ymin=[H*V]*nseg; ymax=[0]*nseg; 
  for i in range(H*V):
    x=i%H; y=i//H; s=seg[i]
    if(x<xmin[s]): xmin[s]=x
    if(y<ymin[s]): ymin[s]=y
    if(x>xmax[s]): xmax[s]=x
    if(y>ymax[s]): ymax[s]=y
  center=np.zeros(nseg,dtype=np.uint32)
  for s in range(nseg):
    y=(ymin[s]+ymax[s])//2; x=(xmin[s]+xmax[s])//2; 
    center[s]=y*H+x
  return(center,nseg)

def select_training_samples_seg(truth,center,H,V,sizex,sizey,porcentaje):
  print('* Select training samples')
  # hacemos una lista con las clases, pero puede haber clases vacias
  nclases=0; nclases_no_vacias=0
  N=len(truth)
  for i in truth:
    if(i>nclases): nclases=i
  print('  nclasses:',nclases)
  lista=[0]*nclases;
  for i in range(nclases):
    lista[i]=[]
  xmin=int(sizex/2); xmax=H-int(math.ceil(sizex/2))
  ymin=int(sizey/2); ymax=V-int(math.ceil(sizey/2))
  for ind in center:
    i=ind//H; j=ind%H;
    if(i<ymin or i>ymax or j<xmin or j>xmax): continue
    if(truth[ind]>0): lista[truth[ind]-1].append(ind)
  for i in range(nclases):
    random.shuffle(lista[i])
  # seleccionamos muestras para train, validacion y test
  print('  Class  # :   total | train |   val |    test')
  train=[]; val=[]; test=[]
  for i in range(nclases):
    # tot0: numero muestras entrenamiento, tot1: validacion 
    if(porcentaje[0]>=1): tot0=porcentaje[0]
    else: tot0=int(porcentaje[0]*len(lista[i]))
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2
    if(tot0<0 and len(lista[i])>0): tot0=1
    if(tot0!=0): nclases_no_vacias+=1
    if(porcentaje[1]>=1): tot1=porcentaje[1]
    else: tot1=int(porcentaje[1]*len(lista[i]))
    if(tot1>=len(lista[i])): tot1=len(lista[i])//2
    if(tot1<1 and len(lista[i])>0): tot1=0
    for j in range(len(lista[i])):
      if(j<tot0): train.append(lista[i][j])
      elif(j<tot0+tot1): val.append(lista[i][j])
      # testeamos los segmentos que no esten en el entrenamiento
      # else: test.append(lista[i][j])
      # testeamos todos los segmentos
      test.append(lista[i][j])
    print('  Class',f'{i+1:2d}',':',f'{len(lista[i]):7d}','|',f'{tot0:5d}','|',
      f'{tot1:5d}','|',f'{len(lista[i])-tot0-tot1:7d}')
  return(train,val,test,nclases,nclases_no_vacias)

#-----------------------------------------------------------------
# PYTORCH - SETS
#-----------------------------------------------------------------

# cogemos muestras sin ground-truth (dadas por el indice samples)
class HyperAllDataset(Dataset):
  def __init__(self,datos,samples,H,V,sizex,sizey):
    self.datos=datos; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(degrees=30)])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    return(patch)

#----------------

# cogemos muestras con ground-truth (dadas por el indice samples)
class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey):
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey;
    self.transform=transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation(degrees=30)])
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    datos=self.datos; truth=self.truth; H=self.H; V=self.V;
    sizex=self.sizex; sizey=self.sizey; 
    x=self.samples[idx]%H; y=int(self.samples[idx]/H)
    patch=select_patch(datos,sizex,sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    # renumeramos porque la red clasifica tambien la clase 0 
    return(patch,truth[self.samples[idx]]-1)

#-----------------------------------------------------------------
# PYTORCH - UTIL
#-----------------------------------------------------------------

# pulsando CNLT-C acabamos el entrenamiento y pasamos a testear
def signal_handler(sig, frame):
  print('\n* Ctrl+C. Exit training')
  global endTrain
  endTrain=True

# For updating learning rate manual
def update_lr(optimizer,lr):    
  for param_group in optimizer.param_groups:
    param_group['lr']=lr

# calcula los promedios de precisiones
def accuracy_mean_deviation(OA,AA,aa):
  n=len(OA); nclases=len(aa[0])
  print('* Means and deviations (%d exp):'%(n))
  # medias
  OAm=0; AAm=0; aam=[0]*nclases;
  for i in range(n):
     OAm+=OA[i]; AAm+=AA[i]
     for j in range(1,nclases): aam[j]+=aa[i][j]
  OAm/=n; AAm/=n
  for j in range(1,nclases): aam[j]/=n
  # desviaciones, usamos la formula que divide entre (n-1)
  OAd=0; AAd=0; aad=[0]*nclases
  for i in range(n):
     OAd+=(OA[i]-OAm)*(OA[i]-OAm); AAd+=(AA[i]-AAm)*(AA[i]-AAm)
     for j in range(1,nclases): aad[j]+=(aa[i][j]-aam[j])*(aa[i][j]-aam[j])
  OAd=math.sqrt(OAd/(n-1)); AAd=math.sqrt(AAd/(n-1))
  for j in range(1,nclases): aad[j]=math.sqrt(aad[j]/(n-1))
  for j in range(1,nclases): print('  Class %02d: %02.02f+%02.02f'%(j,aam[j],aad[j]))
  print('  OA=%02.02f+%02.02f, AA=%02.02f+%02.02f'%(OAm,OAd,AAm,AAd))

# selecciona la funcion de perdida
def select_loss(str_loss, truth, device, n_classes):
  if str_loss == 'CE' or str_loss == 'ce':
    loss=nn.CrossEntropyLoss()
    return loss
  if str_loss == 'balanced_CE' or str_loss == 'balanced_ce':
    truth_no_zeros=[x for x in truth if x != 0]
    truth_no_zeros=np.array(truth_no_zeros)
    all_classes=np.array(range(1, n_classes+1))
    class_weights=np.ones_like(all_classes, dtype=np.float32)
    unique_classes=np.unique(truth_no_zeros)
    calculated_weights=class_weight.compute_class_weight(class_weight='balanced',
       classes=np.unique(truth_no_zeros),y=truth_no_zeros)
    class_weights[np.isin(all_classes, unique_classes)]=calculated_weights
    class_weights=torch.tensor(class_weights,dtype=torch.float)
    class_weights=class_weights.to(device)
    loss=nn.CrossEntropyLoss(weight=class_weights)
    return loss
  if str_loss == 'focal_class':
    loss=FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
    return loss

#-----------------------------------------------------------------
# PYTORCH - MAIN
#-----------------------------------------------------------------

def main(exp):
  print('* ViT exp: '+str(exp))
  time_start=time.time()
  # 1. Device configuration
  cuda=True if torch.cuda.is_available() else False
  print('* Cuda: '+str(cuda))
  device=torch.device('cuda' if cuda else 'cpu')
  if torch.backends.cudnn.is_available():
    print('* Activando CUDNN')
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.beBhmark=True
  # experimentos deterministas o aleatorios
  if(DET==1):
    SEED=0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if(cuda==False):
      torch.use_deterministic_algorithms(True)
      g=torch.Generator(); g.manual_seed(SEED)
    else:
      torch.backends.cudnn.deterministic=True
      torch.backends.cudnn.benchmark=False

  # 2. Load datos
  (datos,H,V,B)=read_raw(DATASET)
  (truth,H1,V1)=read_pgm(GT)
  (seg,H2,V2)=read_seg(SEG)
  # necesitamos los datos en band-vector para hacer convoluciones
  datos=np.transpose(datos,(2,0,1))
  # durante la ejecucion de la red vamos a coger patches de tamano cuadrado
  sizex=SIZEX; sizey=SIZEX 

  # 3. Selection training,testing sets
  # (center,nseg)=seg_center(seg,H,V) # lento, mejor lo cargamos hecho
  (center,H3,V3,nseg)=read_seg_centers(CENTER)
  (train,val,test,nclases,nclases_no_vacias)=select_training_samples_seg(truth,center,H,V,sizex,sizey,SAMPLES)
  dataset_train=HyperDataset(datos,truth,train,H,V,sizex,sizey)
  print('  - train dataset:',len(dataset_train))
  dataset_test=HyperDataset(datos,truth,test,H,V,sizex,sizey)
  print('  - test dataset:',len(dataset_test))
  # Dataloader
  batch_size=BATCH # defecto 100
  train_loader=DataLoader(dataset_train,batch_size,shuffle=True)
  test_loader=DataLoader(dataset_test,batch_size,shuffle=False)
  # Si queremos validacion
  if(len(val)>0):
    dataset_val=HyperDataset(datos,truth,val,H,V,sizex,sizey)
    print('  - val dataset:',len(dataset_val))
    val_loader=DataLoader(dataset_val,batch_size,shuffle=True)
 
  # 4. Hyper parameters
  base_learning_rate=1e-3
  weight_decay=0.05
  mask_ratio=0.75
  warmup_epoch=200
  model_path='/tmp/vit_v3.pt'

  # OJO: no todos estos parametros se usan en este transformer
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

  # 5. Red:
  if(ViTtype==0): model=timm.models.vision_transformer.VisionTransformer(
    img_size=sizex, num_classes=nclases, in_chans=B,
    patch_size=patch_size, depth=depth, num_heads=heads,  
    embed_dim=256, mlp_ratio=4., qkv_bias=False, drop_rate=0.,
    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm).to(device)
  else: model=timm.models.SwinTransformer(
    img_size=sizex, num_classes=nclases, in_chans=B,
    patch_size=patch_size, window_size=16).to(device)

  # 6. Loss, optimizer, and scheduler
  loss_fn=torch.nn.CrossEntropyLoss()
  acc_fn=lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

  optim=torch.optim.AdamW(model.parameters(), lr=base_learning_rate * batch_size / 256, betas=(0.9, 0.95), weight_decay=weight_decay)
  lr_func=lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch /EPOCHS * math.pi) + 1))
  lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

  #------------------------------------------------
  # TRAIN
  #------------------------------------------------

  best_val_acc=0
  step_count=0
  optim.zero_grad()
  for e in range(EPOCHS):
    model.train()
    losses=[]
    acces=[]
    for i,(img,label) in enumerate(train_loader):
      step_count += 1
      img=img.to(device)
      label=label.to(device)
      logits=model(img)
      loss=loss_fn(logits, label)
      acc=acc_fn(logits, label)
      loss.backward()
      optim.step()
      optim.zero_grad()
      losses.append(loss.item())
      acces.append(acc.item())
    lr_scheduler.step()
    avg_train_loss=sum(losses) / len(losses)
    avg_train_acc=sum(acces) / len(acces)

    # si tenemos validacion usamos estas muestras
    if(len(val)>0):
      model.eval()
      with torch.no_grad():
        losses=[]
        acces=[]
        for i,(img,label) in enumerate(val_loader):
          img=img.to(device)
          label=label.to(device)
          logits=model(img)
          loss=loss_fn(logits, label)
          acc=acc_fn(logits, label)
          losses.append(loss.item())
          acces.append(acc.item())
        avg_val_loss=sum(losses) / len(losses)
        avg_val_acc=sum(acces) / len(acces)
    
    if(len(val)>0): print(f'* Epoch {e}, Train loss: %02.04f, Acc: %02.04f, Val. loss: %02.04f, Acc: %02.04f'
       %(avg_train_loss,avg_train_acc,avg_val_loss,avg_val_acc))
    else: print(f'* Epoch {e}, Training loss: %02.04f, Acc: %02.04f'%(avg_train_loss,avg_train_acc))

    if(len(val)>0 and avg_val_acc > best_val_acc):
      best_val_acc=avg_val_acc
      print(f'  (saving best validation model with acc %02.04f at %d epoch)'%(best_val_acc,e))       
      torch.save(model,model_path)

  if(TEST==0): return(sum(acces)/len(acces))

  # 8. Test the model
  print('* Test ViT, exp.%d'%(exp))
  output=np.zeros(H*V,dtype=np.uint8) # mapa de salida de pixels
  # OJO, sale mejor usar el modelo de la ultima-iteracion
  # model=torch.load(model_path, map_location='cpu').to(device)
  model.eval()
  with torch.no_grad():
    correct=0; total=0;
    for(inputs,labels) in test_loader:
      inputs=inputs.to(device)
      labels=labels.to(device)
      outputs=model(inputs)
      (_,predicted)=torch.max(outputs.data,1)
      predicted_cpu=predicted.cpu()
      for i in range(len(predicted_cpu)):
        # queremos que las clases comiencen en 1 en vez de 0
        output[test[total+i]]=np.uint8(predicted_cpu[i]+1)
      total+=labels.size(0)
      if(total%2000==0): print('  Test: %6d/%d'%(total,len(dataset_test)))
  print('* Generating classif.map')
  for i in range(H*V): output[i]=output[center[seg[i]]]
  # eliminamos los centros usados en el entrenamiento
  for i in train: output[i]=0
  for i in val: output[i]=0
  
  # 9. precisiones por segmentos (excluyendo los usados en el entrenamiento)
  correct=0; total=0
  for i in range(len(center)):
    if(output[center[i]]==0): continue
    total+=1
    if(output[center[i]]==truth[center[i]]): correct=correct+1
  acc=100*correct/total;
  print('* Accuracy (segments): %.02f'%(acc))

  # 10. precisiones a nivel de pixel
  correct=0; total=0; AA=0; OA=0
  class_correct=[0]*(nclases+1)
  class_total=[0]*(nclases+1)
  class_aa=[0]*(nclases+1)
  for i in range(len(output)):
    if(output[i]==0 or truth[i]==0): continue
    total+=1; class_total[truth[i]]+=1
    if(output[i]==truth[i]):
      correct+=1
      class_correct[truth[i]]+=1
  for i in range(1,nclases+1):
    if(class_total[i]!=0): class_aa[i]=100*class_correct[i]/class_total[i]
    else: class_aa[i]=0
    AA+=class_aa[i]
  OA=100*correct/total; AA=AA/nclases_no_vacias 
  print('* Accuracy (pixels) exp.%d:'%(exp))
  for i in range(1,nclases+1): print('  Class %02d: %02.02f'%(i,class_aa[i]))
  print('* Accuracy (pixels) exp.%d, OA=%02.02f, AA=%02.02f'%(exp,OA,AA))
  print('  total:',total,'correct:',correct)

  # guardamos la salida
  save_pgm(output,H,V,nclases,'/home/amo/output_vit-'+str(exp)+'.pgm')
  # Save the model checkpoint
  torch.save(model.state_dict(),'/tmp/model_vit-'+str(exp)+'.ckpt')

  time_end=time.time()
  print('* Execution time: %.0f s'%(time_end-time_start))
  print('  lr_base:',base_learning_rate,' imprime aqui otros parametros relevantes')
  return(OA,AA,class_aa)

if __name__=='__main__':
  if(TEST==0): # validacion (se podria hacer una validacion cruzada de varias vias)
    acces=0
    for exp in range(EXP): acces=acces+main(exp)
    print('* ViT TIMM SEG EXP:',EXP,'SAMPLES:',SAMPLES,'EPOCHS:',EPOCHS,'BATCH:',BATCH,'SIZEX:',SIZEX,'AUM:',AUM)
    print('  VAL: %02.02f'%(100*acces/EXP))
  else: # test de 5 experimentos
    OA=[0]*EXP; AA=[0]*EXP; aa=[0]*EXP 
    for exp in range(EXP): (OA[exp],AA[exp],aa[exp])=main(exp)
    if(EXP>1): accuracy_mean_deviation(OA,AA,aa) 
    print('* ViT TIMM SEG EXP:',EXP,'SAMPLES:',SAMPLES,'EPOCHS:',EPOCHS,'BATCH:',BATCH,'SIZEX:',SIZEX,'AUM:',AUM)
