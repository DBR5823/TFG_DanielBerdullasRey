#!/bin/bash

# Directorio de salida
mkdir -p resultadosAumentado_PyTorch

#Nombres de los métodos para el log
METODOS=("Base_Flips" "Base+Rot" "Base+Crop" "Base+Noise" "Base+Rot+Crop" "Base+Rot+Noise" "Base+Crop+Noise")

#Iteramos sobre los datasets (0 a 7)
for d_idx in {0..7}
do
    #Iteramos sobre los métodos de aumento (0 a 6)
    for m_idx in {0..6}
    do
        NOMBRE_METODO=${METODOS[$m_idx]}
        ARCHIVO_LOG="resultadosAumentado_PyTorch/metodo${m_idx}_${NOMBRE_METODO}.log"

        #Limpiamos el archivo si ya existe para guardar únicamente las nuevas estadísticas 
        > "$ARCHIVO_LOG"

        
        echo "**Ejecutando prueba iteración: $d_idx | Método: $NOMBRE_METODO | Inicio: $(date +%T)" | tee -a "$ARCHIVO_LOG**"
        
        # Llamada al script: 
        #ID Dataset
        #Sampler (1 para activarlo)
        #ID Método Aumento
        python3 -u cnn21_pruebasParalelasBASE_PyTorch.py "$d_idx" 1 "$m_idx" 2>&1 | tee -a "$ARCHIVO_LOG"
        
        echo "FINALIZACIÓN prueba iteración: $d_idx | Método: $NOMBRE_METODO | $(date +%T)" | tee -a "$ARCHIVO_LOG"
        echo "********************************************" | tee -a "$ARCHIVO_LOG"
        sleep 3
    done
done

echo "Experimento finalizado