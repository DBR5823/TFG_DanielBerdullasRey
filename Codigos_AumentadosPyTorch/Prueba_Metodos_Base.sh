#!/bin/bash
#SBATCH --job-name=EXP_CNN_PYTORCH
#SBATCH -w ctgpgpu5                  # Fuerza el nodo actual
#SBATCH --gres=gpu:2                 # Reserva las 2 GPUs
#SBATCH --cpus-per-task=8            # Reserva exactamente 8 núcleos de CPU
#SBATCH --partition=gpu              # Partición de GPUs del CITIUS
#SBATCH --output=log_TOTAL_%j.out    # Archivo donde verás los "echo" y resultados
#SBATCH --error=log_TOTAL_%j.err     # Archivo para errores
#SBATCH --time=60:00:00              # Duración estimada (aumentada por los nuevos métodos)


source /home/daniel.berdullas/TFG/venv/bin/activate

# Directorio de salida
mkdir -p resultadosAumentado_PyTorch

# Nombres de los métodos actualizados (del 0 al 10, según tu nuevo código Python)
METODOS=(
    "Sin_Aumentado" 
    "Rotacion" 
    "Zoom_Simetrico" 
    "Rotacion_Zoom" 
    "Ruido_Gaussiano" 
    "Ruido_Espectral" 
    "Ruido_Gaussiano_Espectral" 
    "Iluminacion_Aleatoria" 
    "Eliminar_Bandas" 
    "Iluminacion_Eliminar_Bandas" 
    "Borrado_Aleatorio"
    "Rotacion_Borrado_Aleatorio"                  
    "Rotacion_Zoom_Borrado_Aleatorio"
    "Rotacion_Ruido_Espectral"
    "Rotacion_Zoom_Ruido_Espectral"
    "Ruido_Espectral_Borrado_Aleatorio"
)

# 1. INICIALIZACIÓN DE LOGS
# Vaciamos (o creamos) los 15 archivos de log antes de empezar las pruebas.
# Así nos aseguramos de no arrastrar datos de ejecuciones anteriores.
for m_idx in {0..15}
do
    NOMBRE_METODO=${METODOS[$m_idx]}
    > "resultadosAumentado_PyTorch/metodo_${m_idx}_${NOMBRE_METODO}.log"
done

# 2. EJECUCIÓN DE LOS EXPERIMENTOS
# Iteramos sobre los datasets (0 a 7)
for d_idx in {0..7}
do
    # Iteramos sobre los métodos de aumento (0 a 15)
    for m_idx in {0..15}
    do
        NOMBRE_METODO=${METODOS[$m_idx]}
        ARCHIVO_LOG="resultadosAumentado_PyTorch/metodo_${m_idx}_${NOMBRE_METODO}.log"
        
        echo "**Ejecutando dataset: $d_idx | Método: $NOMBRE_METODO | Inicio: $(date +%T)**" | tee -a "$ARCHIVO_LOG"
        
        # Llamada al script: 
        # $1 -> ID Dataset ($d_idx)
        # $2 -> Sampler (1 para activarlo)
        # $3 -> ID Método Aumento ($m_idx)
        python3 -u cnn21_pruebasParalelasBASE_PyTorch.py "$d_idx" 1 "$m_idx" 2>&1 | tee -a "$ARCHIVO_LOG"
        
        echo "FINALIZACIÓN dataset: $d_idx | Método: $NOMBRE_METODO | $(date +%T)" | tee -a "$ARCHIVO_LOG"
        echo "********************************************" | tee -a "$ARCHIVO_LOG"
        
        sleep 3
    done
done

echo "Experimentos finalizados correctamente."