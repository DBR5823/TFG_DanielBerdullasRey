#!/bin/bash

#Nombre del archivo donde se guardarán los resultados de las pruebas
ARCHIVO_LOG="resultadosPruebasVIT/resultados_VIT+CUTMIX+FMIX.log"

#Limpiamos el archivo si ya existe para guardar únicamente las nuevas estadísticas 
> "$ARCHIVO_LOG"

echo "********** Iniciando experimentos VIT+CUTMIX+FMIX con aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"

# 1. Ejecución inicial sin argumentos para ejecutar la prueba sobre el dataset de Oitaven
echo "Ejecutando: prueba inicial CUTMIX+FMIX (sobre Oitaven)" | tee -a "$ARCHIVO_LOG"
python3 -u vit_CutMix+FMIX_pruebasParalelas.py 0 1 2>&1 | tee -a "$ARCHIVO_LOG"
sleep 2

echo "------------------------------------------" | tee -a "$ARCHIVO_LOG"

# 2. Bucle del 1 al 7 para ejecutar la prueba sobre todos los datasets
for i in {1..7}
do
    echo "Ejecutando prueba iteración: $i..." | tee -a "$ARCHIVO_LOG"
    # Ejecutamos el comando, redirigimos errores (stderr) al mismo lugar y usamos tee
    python3 -u vit_CutMix+FMIX_pruebasParalelas.py "$i" 1 2>&1 | tee -a "$ARCHIVO_LOG"
    sleep 2
    echo "Prueba iteracion $i finalizada." | tee -a "$ARCHIVO_LOG"
    echo "********************************************" | tee -a "$ARCHIVO_LOG"
done

echo "********** Se han finalizado todas las pruebas de VIT+CUTMIX+FMIX sin aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"
