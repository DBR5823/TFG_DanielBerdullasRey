#!/usr/bin/env python3
import math
import os
import numpy as np
import torch
import torchvision.utils as vutils

# IMPORTACIÓN DE TU IMPLEMENTACIÓN ORIGINAL
from implementations.torchbearer_implementation import FMix

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
    patch = datos[y1:y2, x1:x2, :]
    return patch.permute(2, 0, 1)

def save_raw(output, H, V, B, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print('No puedo abrir ', filename)
        return
    sizes = np.array([B, H, V], dtype=np.uint32)
    output = output.detach().cpu().numpy()
    if len(output.shape) == 3:
        output = np.transpose(output, (1, 2, 0))
    output_flat = (output * 255).astype(np.uint32) # Forzamos a uint32 para tu visualizador
    final_data = np.concatenate([sizes, output_flat.flatten()])
    final_data.tofile(f)
    f.close()

def save_patch(datos, sizex, sizey, B, filename):
    save_raw(datos.clone().cpu(), sizex, sizey, B, filename)

if __name__ == '__main__':
    DATASET = '/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.raw'
    sizex, sizey = 32, 32
    carpeta_salida = "figuras_fmix"
    os.makedirs(carpeta_salida, exist_ok=True)

    # Parámetros originales
    fmix_alpha = 1.0
    fmix_decay = 2.0
    fmix_soft = 0.0

    print("Leyendo Dataset...")
    datos_raw, H, V, B = read_raw(DATASET)
    
    # Seleccionamos dos parches con gran contraste (p.ej. agua vs vegetación)
    # Ajusta estas coordenadas si no ves contraste
    patch_a = select_patch(datos_raw, sizex, sizey, 4100, 3200) 
    patch_b = select_patch(datos_raw, sizex, sizey, 5500, 5500) 

    # Crea un batch de solo 2 parches distintos
    inputs = torch.stack([patch_a, patch_b]) 

    # Inicializamos FMix
    fmix_util = FMix(size=(sizex, sizey), alpha=fmix_alpha, decay_power=fmix_decay, max_soft=fmix_soft)

    # --- TRUCO PARA TESTEO ---
    # Forzamos los índices para que el parche 0 se mezcle con el 1
    # En lugar de dejarlo al azar, le decimos: mezcla el primero con el segundo.
    fmix_util.index = torch.tensor([1, 0]) 
    # -------------------------

    inputs_mixed = fmix_util(inputs)
    mask = fmix_util.mask 
    indices = fmix_util.index
    lam = fmix_util.lam

    print(f"Índices de mezcla: {fmix_util.index}")
    print(f"Mezclando parche 0 con parche {fmix_util.index[0]}")

    # Guardamos los resultados
    for j in range(1):
        # El parche original A
        save_patch(inputs[0], sizex, sizey, B, os.path.join(carpeta_salida, f'parche_{0}_orig.raw'))

        # El parche original B
        save_patch(inputs[1], sizex, sizey, B, os.path.join(carpeta_salida, f'parche_{1}_orig.raw'))
        
        # El parche mezclado resultante
        save_patch(inputs_mixed[j], sizex, sizey, B, os.path.join(carpeta_salida, f'parche_{j}_MIXED.raw'))

        # Guardamos la máscara generada
        vutils.save_image(mask, os.path.join(carpeta_salida, f'parche_{j}_mascara.png'))

    print(f"\nLambda obtenido: {lam:.4f}")