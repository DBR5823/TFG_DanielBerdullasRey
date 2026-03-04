#!/bin/bash

#Nombre del archivo donde se guardarán los resultados de las pruebas
ARCHIVO_LOG="resultadosPruebas/resultados_CNN_Basica.log"

#Limpiamos el archivo si ya existe para guardar únicamente las nuevas estadísticas 
> "$ARCHIVO_LOG"

echo "********** Iniciando experimentos CNN Básica sin aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"

# 1. Ejecución inicial sin argumentos para ejecutar la prueba sobre el dataset de Oitaven
echo "Ejecutando: prueba inicial CNN Básica (sobre Oitaven)" | tee -a "$ARCHIVO_LOG"
python3 -u cnn21_pruebasParalelas.py 0 0 2>&1 | tee -a "$ARCHIVO_LOG"

echo "------------------------------------------" | tee -a "$ARCHIVO_LOG"

# 2. Bucle del 1 al 7 para ejecutar la prueba sobre todos los datasets
for i in {1..7}
do
    echo "Ejecutando prueba iteración: $i..." | tee -a "$ARCHIVO_LOG"
    # Ejecutamos el comando, redirigimos errores (stderr) al mismo lugar y usamos tee
    python3 -u cnn21_pruebasParalelas.py "$i" 0 2>&1 | tee -a "$ARCHIVO_LOG"
    echo "Prueba iteracion $i finalizada." | tee -a "$ARCHIVO_LOG"
    echo "********************************************" | tee -a "$ARCHIVO_LOG"
done

echo "********** Se han finalizado todas las pruebas de CNN Básica sin aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"


#Nombre del archivo donde se guardarán los resultados de las pruebas
ARCHIVO_LOG="resultadosPruebasAumentado/resultados_CNN_Basica_aumentado.log"

#Limpiamos el archivo si ya existe para guardar únicamente las nuevas estadísticas 
> "$ARCHIVO_LOG"

echo "********** Iniciando experimentos CNN Básica con aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"

# 1. Ejecución inicial sin argumentos para ejecutar la prueba sobre el dataset de Oitaven
echo "Ejecutando: prueba inicial CNN Básica (sobre Oitaven)" | tee -a "$ARCHIVO_LOG"
python3 -u cnn21_pruebasParalelas.py 0 1 2>&1 | tee -a "$ARCHIVO_LOG"

echo "------------------------------------------" | tee -a "$ARCHIVO_LOG"

# 2. Bucle del 1 al 7 para ejecutar la prueba sobre todos los datasets
for i in {1..7}
do
    echo "Ejecutando prueba iteración: $i..." | tee -a "$ARCHIVO_LOG"
    # Ejecutamos el comando, redirigimos errores (stderr) al mismo lugar y usamos tee
    python3 -u cnn21_pruebasParalelas.py "$i" 1 2>&1 | tee -a "$ARCHIVO_LOG"
    echo "Prueba iteracion $i finalizada." | tee -a "$ARCHIVO_LOG"
    echo "********************************************" | tee -a "$ARCHIVO_LOG"
done

echo "********** Se han finalizado todas las pruebas de CNN Básica con aumentado: $(date +%T) **********" | tee -a "$ARCHIVO_LOG"
