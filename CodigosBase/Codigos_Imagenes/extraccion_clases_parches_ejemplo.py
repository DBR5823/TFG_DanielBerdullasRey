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

def read_pgm(fichero):
    try:
        pgmf = open(fichero, "rb")
    except IOError:
        print('No puedo abrir ', fichero)
        return None, 0, 0, 0, []

    # Verificar formato P5
    assert pgmf.readline().decode() == 'P5\n'

    # Leer y GUARDAR los comentarios
    comentarios = []
    line = pgmf.readline().decode()
    while line[0] == '#':
        comentarios.append(line)
        line = pgmf.readline().decode()
    
    # Dimensiones
    H, V = line.split()
    H, V = int(H), int(V)

    # Profundidad/Valor máximo
    depth = int(pgmf.readline().decode())
    assert depth <= 255

    # Lectura 
    raster = np.fromfile(pgmf, dtype=np.uint8, count=H*V)
    pgmf.close()

    raster = raster.reshape(V, H)
    
    # Ahora devolvemos también los comentarios extraídos
    return torch.LongTensor(raster), H, V, depth, comentarios

def select_patch(datos, sizex, sizey, x, y, es_filtrado_bandas=True):
    x1 = x - int(sizex/2); x2 = x + int(math.ceil(sizex/2))
    y1 = y - int(sizey/2); y2 = y + int(math.ceil(sizey/2))
    
    patch = datos[y1:y2, x1:x2]
    
    if es_filtrado_bandas:
        return patch.permute(2, 0, 1)
    return patch

def save_raw(output, H, V, B, filename):
    try:
        f = open(filename, "wb")
    except IOError:
        print('No se puede abrir ', filename)
        return
    sizes = np.array([B, H, V], dtype=np.uint32)
    output = output.detach().cpu().numpy()
    if len(output.shape) == 3:
        output = np.transpose(output, (1, 2, 0))
    output_flat = (output * 255).astype(np.uint32) 
    final_data = np.concatenate([sizes, output_flat.flatten()])
    final_data.tofile(f)
    f.close()

def save_pgm(output, H, V, depth, comentarios, filename):
    """Guarda el PGM inyectando los comentarios originales para conservar colores y clases"""
    try:
        f = open(filename, "wb")
    except IOError:
        print('No se puede abrir ', filename)
        return
    
    # 1. Escribir el Magic Number P5
    f.write(b"P5\n")
    
    # 2. Inyectar todos los comentarios originales (colores/clases)
    for comentario in comentarios:
        f.write(comentario.encode())
        
    # 3. Escribir dimensiones y profundidad
    f.write(f"{H} {V}\n{depth}\n".encode())
    
    # 4. Escribir los píxeles
    output_np = output.detach().cpu().numpy().astype(np.uint8)
    output_np.tofile(f)
    f.close()

if __name__ == '__main__':
    # Rutas
    DATASET_RAW = '/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.raw'
    DATASET_PGM = '/home/dbr/Escritorio/TFG/cnn21/datosEntrada/oitaven/oitaven_river.pgm'
    
    sizex, sizey = 32, 32
    carpeta_salida = "parches_extraidos"
    os.makedirs(carpeta_salida, exist_ok=True)

    print("Leyendo Dataset RAW...")
    datos_raw, H_raw, V_raw, B = read_raw(DATASET_RAW)
    
    print("Leyendo Dataset PGM (Clasificación)...")
    # Capturamos la nueva variable 'comentarios'
    datos_pgm, H_pgm, V_pgm, depth, comentarios_pgm = read_pgm(DATASET_PGM)
    
    assert H_raw == H_pgm and V_raw == V_pgm, "Las dimensiones del RAW y el PGM no coinciden."

    coordenadas = [
        {"x": 4500, "y": 3000, "id": 0},
        {"x": 4450, "y": 2700, "id": 1}
    ]

    for coord in coordenadas:
        x, y, idx = coord["x"], coord["y"], coord["id"]
        print(f"\nExtrayendo Parche {idx} en posición ({x}, {y})...")
        
        # RAW
        patch_raw = select_patch(datos_raw, sizex, sizey, x, y, es_filtrado_bandas=True)
        nombre_raw = os.path.join(carpeta_salida, f'parche_{idx}_orig.raw')
        save_raw(patch_raw, sizex, sizey, B, nombre_raw)
        print(f"  -> Guardado RAW: {nombre_raw}")
        
        # PGM
        patch_pgm = select_patch(datos_pgm, sizex, sizey, x, y, es_filtrado_bandas=False)
        nombre_pgm = os.path.join(carpeta_salida, f'parche_{idx}_clasificacion.pgm')
        
        # Pasamos los comentarios a la función de guardado
        save_pgm(patch_pgm, sizex, sizey, depth, comentarios_pgm, nombre_pgm)
        print(f"  -> Guardado PGM: {nombre_pgm}")

    print("\n¡Proceso finalizado con éxito!")