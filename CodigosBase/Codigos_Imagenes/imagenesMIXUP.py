#!/usr/bin/env python3
import math
import os
import numpy as np
import torch

def read_raw(fichero):
    (B, H, V) = np.fromfile(fichero, count=3, dtype=np.uint32)
    datos = np.fromfile(fichero, count=B*H*V, offset=3*4, dtype=np.int32).astype(np.float32)
    
    d_min, d_max = datos.min(), datos.max()
    datos = (datos - d_min) / (d_max - d_min)
    
    datos = datos.reshape(V, H, B)
    return torch.FloatTensor(datos), H, V, B

def select_patch(datos, sizex, sizey, x, y):
    x1 = x - int(sizex/2); x2 = x + int(math.ceil(sizex/2))
    y1 = y - int(sizey/2); y2 = y + int(math.ceil(sizey/2))
    # PyTorch espera (C, H, W), los datos vienen (V, H, B)
    # Hacemos el slice y permutamos a (B, V, H)
    patch = datos[y1:y2, x1:x2, :]
    return patch.permute(2, 0, 1)

def save_raw(output, H, V, B, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print('No se puede abrir ', filename)
        return
    
    sizes = np.array([B, H, V], dtype=np.uint32)
    # Asegurar orden (B, H, V) para el guardado compatible con el lector
    output = output.detach().cpu().numpy()
    # Si viene de PyTorch (B, V, H), trasponemos a (V, H, B) para aplanar correctamente
    if len(output.shape) == 3:
        output = np.transpose(output, (1, 2, 0))
    
    output_flat = output.flatten().astype(np.uint32)
    final_data = np.concatenate([sizes, output_flat])
    final_data.tofile(f)
    f.close()

def save_patch(datos, sizex, sizey, B, filename):
    datos_copy = datos.clone().cpu()
    if datos_copy.max() <= 1.0:
        datos_copy = datos_copy * 255
    save_raw(datos_copy, sizex, sizey, B, filename)


#Función MIXUP para aplicar la mezcla MIXUP de dos patches
def aplicar_mixup_visual(patch_a, patch_b, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    #Mezcla píxel a píxel
    patch_mixed = lam * patch_a + (1 - lam) * patch_b
    
    return patch_mixed, lam


if __name__ == '__main__':
    DATASET = '/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.raw'
    sizex, sizey = 32, 32
    carpeta_salida = "figuras_mixup"
    os.makedirs(carpeta_salida, exist_ok=True)

    print("Leyendo Dataset...")
    datos_raw, H, V, B = read_raw(DATASET)
    

    # Parche A y B (COORDENADAS ACTUALIZADAS)
    patch_a = select_patch(datos_raw, sizex, sizey, 4500, 3000) 
    patch_b = select_patch(datos_raw, sizex, sizey, 4450, 2700) 
    
    save_patch(patch_a, sizex, sizey, B, os.path.join(carpeta_salida, "0_Original_A.raw"))
    save_patch(patch_b, sizex, sizey, B, os.path.join(carpeta_salida, "0_Original_B.raw"))


    configuraciones = [0.2, 1.0, 10.0]
    
    print("Generando mezclas Mixup...")
    for alpha in configuraciones:
        for i in range(2):
            mezcla, lam = aplicar_mixup_visual(patch_a, patch_b, alpha=alpha)
            nombre = f"Mixup_alpha{alpha}_ej{i}.raw"
            save_patch(mezcla, sizex, sizey, B, os.path.join(carpeta_salida, nombre))
            print(f"    Guardado {nombre} | Lambda: {lam:.4f} (A al {lam*100:.1f}%)")