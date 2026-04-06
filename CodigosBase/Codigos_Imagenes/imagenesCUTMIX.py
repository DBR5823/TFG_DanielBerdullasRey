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
        print('No puedo abrir ', filename)
        return
    
    sizes = np.array([B, H, V], dtype=np.uint32)
    # Asegurar orden (B, H, V) para el guardado compatible con tu lector
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

def aplicar_cutmix_visual(patch_a, patch_b, alpha=1.0):
    """
    Versión simplificada de CutMix para generar un solo ejemplo visual.
    """
    W, H = patch_a.size(2), patch_a.size(1)
    
    # Generar proporción de mezcla (lambda)
    lam = np.random.beta(alpha, alpha)
    
    # Calcular dimensiones del recorte
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Centro aleatorio
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Coordenadas del cuadro (clipping)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Crear mezcla
    patch_mixed = patch_a.clone()
    patch_mixed[:, bby1:bby2, bbx1:bbx2] = patch_b[:, bby1:bby2, bbx1:bbx2]
    
    return patch_mixed, (bbx1, bby1, bbx2, bby2)

# --- MAIN ---
if __name__ == '__main__':
    DATASET = '/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.raw'
    sizex, sizey = 32, 32
    carpeta_salida = "figuras_cutmix"
    os.makedirs(carpeta_salida, exist_ok=True)

    print("Leyendo Dataset...")
    datos_raw, H, V, B = read_raw(DATASET)
    
    # Buscamos dos parches diferentes para mezclar
    # Parche A: Zona del río (brillante/agua)
    # Parche B: Zona de vegetación (más oscura o distinta firma)
    print("Extrayendo parches de control...")
    patch_a = select_patch(datos_raw, sizex, sizey, 4100, 3200) # Coordenadas arbitrarias
    patch_b = select_patch(datos_raw, sizex, sizey, 5500, 5500) # Coordenadas alejadas
    
    # Guardar originales para comparar
    save_patch(patch_a, sizex, sizey, B, os.path.join(carpeta_salida, "0_Original_A.raw"))
    save_patch(patch_b, sizex, sizey, B, os.path.join(carpeta_salida, "0_Original_B.raw"))

    # Aplicar CutMix con diferentes Alphas
    configuraciones = [0.2, 1.0, 10.0]
    
    print("Generando mezclas CutMix...")
    for alpha in configuraciones:
        # Generamos 2 ejemplos por cada alpha
        for i in range(2):
            mezcla, coords = aplicar_cutmix_visual(patch_a, patch_b, alpha=alpha)
            nombre = f"CutMix_alpha{alpha}_ej{i}.raw"
            save_patch(mezcla, sizex, sizey, B, os.path.join(carpeta_salida, nombre))
            print(f"    Guardado {nombre} | Recorte en: {coords}")

