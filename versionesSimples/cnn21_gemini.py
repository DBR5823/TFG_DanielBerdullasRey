#!/usr/bin/env python3
import math, random, struct, signal, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torchvision.transforms as transforms
import torchvision.utils as vutils

# ### FMIX: Importación de la implementación
from implementations.lightning import FMix

EXP=5      
EPOCHS=100 
SAMPLES=[0.15,0.05] 
BATCH=100  
ADA=3  
AUM=1  
DET=0  
TEST=1 
ALL=0  

#Rutas de archivos
#Dataset: dataset original, contiene la información obtenida por el dron (cada píxel tiene un cierto número de bandas con datos en cada una)
DATASET='/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.raw'
#GT: Etiquetas de cada segmento, son las etiquetas reales correspondientes a cada segmento, los segmentos son de 32 x 32 píxeles centrados en un centro.
GT='/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.pgm'
#SEG: segmentación, cada píxel tiene el ID del segmento al que pertenece
SEG='/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/seg_oitaven_wp.raw'
#CENTER: centros de los segmentos, contiene los índices de cada píxel correspondiente al centro de cada segmento.
CENTER='/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/seg_oitaven_wp_centers.raw'

#-----------------------------------------------------------------
# FUNCIONES DE LECTURA (Sin cambios)
#-----------------------------------------------------------------

def read_raw(fichero):
  (B,H,V)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=B*H*V,offset=3*4,dtype=np.int32)
  datos=preprocessing.minmax_scale(datos)
  datos=datos.reshape(V,H,B)
  datos=torch.FloatTensor(datos)
  return(datos,H,V,B)

def read_seg(fichero):
  (H,V)=np.fromfile(fichero,count=2,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=2*4,dtype=np.uint32)
  return(datos,H,V)

def read_seg_centers(fichero):
  (H,V,nseg)=np.fromfile(fichero,count=3,dtype=np.uint32)
  datos=np.fromfile(fichero,count=H*V,offset=3*4,dtype=np.uint32)
  return(datos,H,V,nseg)

def save_pgm(output,H,V,nclases,filename):
  try:
    f=open(filename,"wb")
    cadena='P5\n'+str(H)+' '+str(V)+'\n'+str(nclases)+'\n'
    f.write(bytes(cadena,'utf-8'))
    f.write(output)
    f.close()
  except IOError: print('No puedo abrir ',filename)

def select_patch(datos,sizex,sizey,x,y):
  x1=x-int(sizex/2); x2=x+int(math.ceil(sizex/2))     
  y1=y-int(sizey/2); y2=y+int(math.ceil(sizey/2))
  patch=datos[:,y1:y2,x1:x2]
  return(patch)

def read_pgm(fichero):
  pgmf=open(fichero,"rb")
  assert pgmf.readline().decode()=='P5\n'
  line=pgmf.readline().decode()
  while(line[0]=='#'): line=pgmf.readline().decode()
  (H,V)=line.split(); H=int(H); V=int(V)
  depth=int(pgmf.readline().decode())
  raster=[]
  for i in range(H*V): raster.append(ord(pgmf.read(1)))
  return(raster,H,V)

def select_training_samples_seg(truth,center,H,V,sizex,sizey,porcentaje):
  nclases=0; nclases_no_vacias=0
  for i in truth:
    if(i>nclases): nclases=i
  lista=[[] for _ in range(nclases)]
  xmin=int(sizex/2); xmax=H-int(math.ceil(sizex/2))
  ymin=int(sizey/2); ymax=V-int(math.ceil(sizey/2))
  for ind in center:
    i=ind//H; j=ind%H
    if(i<ymin or i>ymax or j<xmin or j>xmax): continue
    if(truth[ind]>0): lista[truth[ind]-1].append(ind)
  for i in range(nclases): random.shuffle(lista[i])
  train=[]; val=[]; test=[]
  for i in range(nclases):
    tot0=int(porcentaje[0]*len(lista[i])) if porcentaje[0]<1 else porcentaje[0]
    if(tot0>=len(lista[i])): tot0=len(lista[i])//2
    if(tot0<=0 and len(lista[i])>0): tot0=1
    if(tot0!=0): nclases_no_vacias+=1
    tot1=int(porcentaje[1]*len(lista[i])) if porcentaje[1]<1 else porcentaje[1]
    if(tot1>=len(lista[i])-tot0): tot1=(len(lista[i])-tot0)//2
    for j in range(len(lista[i])):
      if(j<tot0): train.append(lista[i][j])
      elif(j<tot0+tot1): val.append(lista[i][j])
      test.append(lista[i][j])
  return(train,val,test,nclases,nclases_no_vacias)

#-----------------------------------------------------------------
# DATASETS
#-----------------------------------------------------------------

class HyperDataset(Dataset):
  def __init__(self,datos,truth,samples,H,V,sizex,sizey):
    self.datos=datos; self.truth=truth; self.samples=samples
    self.H=H; self.V=V; self.sizex=sizex; self.sizey=sizey
    self.transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()])
    
  def __len__(self): return len(self.samples)

  def __getitem__(self,idx):
    x=self.samples[idx]%self.H; y=int(self.samples[idx]/self.H)
    patch=select_patch(self.datos,self.sizex,self.sizey,x,y)
    if(AUM==1): patch=self.transform(patch)
    return(patch, self.truth[self.samples[idx]]-1)

#-----------------------------------------------------------------
# MODELO
#-----------------------------------------------------------------

class CNN21(nn.Module):
  def __init__(self,N1,N2,N3,N4,N5,D1,D2):
    super(CNN21,self).__init__()
    self.layer1=nn.Sequential(
      nn.Conv2d(N1,N2,kernel_size=3,stride=1,padding=2),
      nn.BatchNorm2d(N2), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D1))
    self.layer2=nn.Sequential(
      nn.Conv2d(N2,N3,kernel_size=5,stride=1,padding=2),
      nn.BatchNorm2d(N3), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2,stride=D2))
    self.fc=nn.Linear(N4,N5)
      
  def forward(self,x):
    out=self.layer1(x)
    out=self.layer2(out)
    out=out.reshape(out.size(0),-1)
    return self.fc(out)

#-----------------------------------------------------------------
# MAIN
#-----------------------------------------------------------------

def main(exp):
  time_start=time.time()
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # 2. Load datos
  (datos,H,V,B)=read_raw(DATASET)
  (truth,H1,V1)=read_pgm(GT)
  (seg,H2,V2)=read_seg(SEG)
  datos=np.transpose(datos,(2,0,1))
  sizex=32; sizey=32 

  # 3. Selection sets
  (center,H3,V3,nseg)=read_seg_centers(CENTER)
  (train,val,test,nclases,nclases_no_vacias)=select_training_samples_seg(truth,center,H,V,sizex,sizey,SAMPLES)
  
  train_loader=DataLoader(HyperDataset(datos,truth,train,H,V,sizex,sizey),BATCH,shuffle=True)
  test_loader=DataLoader(HyperDataset(datos,truth,test,H,V,sizex,sizey),BATCH,shuffle=False)
  val_loader=DataLoader(HyperDataset(datos,truth,val,H,V,sizex,sizey),BATCH,shuffle=False) if len(val)>0 else None

  # 5. Red
  N1=B; D1=2; H1=sizex; N2=16; H2=int(H1/D1)
  N3=32; D2=2; H3=int(H2/D2)
  model=CNN21(N1,N2,N3,H3*H3*N3,nclases,D1,D2).to(device)

  # ### FMIX: Inicialización
  # El objeto fmix generará las máscaras de Fourier para el tamaño de parche 32x32
  fmix = FMix(alpha=1.0, decay_power=3.0, size=(sizex, sizey))

  # 6. Loss & Optimizer
  criterion=nn.CrossEntropyLoss()
  optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS) if ADA==3 else None

  # 7. Train
  print(f'* Train CNN21 + FMix, exp.{exp}')
  for epoch in range(EPOCHS):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      
      # ### FMIX: Aplicación de mezcla de imágenes
      inputs = fmix(inputs)
      
      outputs = model(inputs)
      
      # ### FMIX: Cálculo de pérdida con mezcla de etiquetas
      loss = fmix.loss(outputs, labels)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
       
    if(epoch%10==0 or epoch==EPOCHS-1):
      print(f'  Epoch: {epoch}/{EPOCHS}, Loss: {loss.item():.4f}')
    
    if scheduler: scheduler.step()

  # 8. Test (Sin FMix)
  model.eval()
  output=np.zeros(H*V,dtype=np.uint8)
  with torch.no_grad():
    total=0
    for(inputs,labels) in test_loader:
      inputs=inputs.to(device)
      outputs=model(inputs)
      predicted=outputs.argmax(1).cpu()
      for i in range(len(predicted)):
        output[test[total+i]]=np.uint8(predicted[i]+1)
      total+=labels.size(0)

  # 9-10. Métricas (Resumen)
  for i in range(H*V): output[i]=output[center[seg[i]]]
  for i in train+val: output[i]=0
  
  correct=0; total_px=0; class_correct=[0]*(nclases+1); class_total=[0]*(nclases+1)
  for i in range(len(output)):
    if(output[i]==0 or truth[i]==0): continue
    total_px+=1; class_total[truth[i]]+=1
    if(output[i]==truth[i]):
      correct+=1; class_correct[truth[i]]+=1
  
  OA = 100*correct/total_px if total_px>0 else 0
  AA = np.mean([100*class_correct[i]/class_total[i] for i in range(1, nclases+1) if class_total[i]>0])
  print(f'* OA={OA:.02f}, AA={AA:.02f}')
  
  return(OA, AA, [])

if __name__=='__main__':
  OA_list=[]; AA_list=[]
  for exp in range(EXP):
    oa, aa, _ = main(exp)
    OA_list.append(oa); AA_list.append(aa)
  print(f'\n* Final Results: OA={np.mean(OA_list):.2f}, AA={np.mean(AA_list):.2f}')
