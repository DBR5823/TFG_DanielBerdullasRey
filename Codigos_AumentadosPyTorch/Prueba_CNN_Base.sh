#!/bin/bash
#SBATCH --job-name=EXP_CNN_TRANSFORM_DANIEL
#SBATCH -w ctgpgpu5                  # Fuerza el nodo actual
#SBATCH --gres=gpu:2                 # Reserva las 2 GPUs
#SBATCH --partition=gpu              # Partición de GPUs del CITIUS
#SBATCH --output=log_TOTAL_%j.out    # Archivo donde verás los "echo" y resultados
#SBATCH --error=log_TOTAL_%j.err     # Archivo para errores
#SBATCH --time=20:00:00              #Duración estimada
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