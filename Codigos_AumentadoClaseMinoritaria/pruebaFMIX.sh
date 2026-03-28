#!/bin/bash

#Nombre del archivo donde se guardarán los resultados de las pruebas
ARCHIVO_LOG="resultadosPruebas/resultados_FMIX.log"

#Limpiamos el archivo si ya existe para guardar únicamente las nuevas estadísticas 
> "$ARCHIVO_LOG"

echo "********** Iniciando experimentos FMIX sin aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"

# 1. Ejecución inicial sin argumentos para ejecutar la prueba sobre el dataset de Oitaven
echo "Ejecutando: prueba inicial FMIX (sobre Oitaven)" | tee -a "$ARCHIVO_LOG"
python3 -u cnn_21_FMIX_PARALELO.py 0 0 2>&1 | tee -a "$ARCHIVO_LOG"
sleep 2

echo "------------------------------------------" | tee -a "$ARCHIVO_LOG"

# 2. Bucle del 1 al 7 para ejecutar la prueba sobre todos los datasets
for i in {1..7}
do
    echo "Ejecutando prueba iteración: $i..." | tee -a "$ARCHIVO_LOG"
    # Ejecutamos el comando, redirigimos errores (stderr) al mismo lugar y usamos tee
    python3 -u cnn_21_FMIX_PARALELO.py "$i" 0 2>&1 | tee -a "$ARCHIVO_LOG"
    sleep 2
    echo "Prueba iteracion $i finalizada." | tee -a "$ARCHIVO_LOG"
    echo "********************************************" | tee -a "$ARCHIVO_LOG"
done

echo "********** Se han finalizado todas las pruebas de FMIX sin aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"


#Nombre del archivo donde se guardarán los resultados de las pruebas
ARCHIVO_LOG="resultadosPruebasAumentado/resultados_FMIX_aumentado.log"

#Limpiamos el archivo si ya existe para guardar únicamente las nuevas estadísticas 
> "$ARCHIVO_LOG"

echo "********** Iniciando experimentos FMIX con aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"

# 1. Ejecución inicial sin argumentos para ejecutar la prueba sobre el dataset de Oitaven
echo "Ejecutando: prueba inicial FMIX (sobre Oitaven)" | tee -a "$ARCHIVO_LOG"
python3 -u cnn_21_FMIX_PARALELO.py 0 1 2>&1 | tee -a "$ARCHIVO_LOG"
sleep 2

echo "------------------------------------------" | tee -a "$ARCHIVO_LOG"

# 2. Bucle del 1 al 7 para ejecutar la prueba sobre todos los datasets
for i in {1..7}
do
    echo "Ejecutando prueba iteración: $i..." | tee -a "$ARCHIVO_LOG"
    # Ejecutamos el comando, redirigimos errores (stderr) al mismo lugar y usamos tee
    python3 -u cnn_21_FMIX_PARALELO.py "$i" 1 2>&1 | tee -a "$ARCHIVO_LOG"
    sleep 2
    echo "Prueba iteracion $i finalizada." | tee -a "$ARCHIVO_LOG"
    echo "********************************************" | tee -a "$ARCHIVO_LOG"
done

echo "********** Se han finalizado todas las pruebas de FMIX con aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"
