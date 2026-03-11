#!/bin/bash
#SBATCH --job-name=EXP_CNN_TRANSFORM_DANIEL
#SBATCH -w ctgpgpu5                  # Fuerza el nodo actual
#SBATCH --gres=gpu:2                 # Reserva las 2 GPUs
#SBATCH --partition=gpu              # Partición de GPUs del CITIUS
#SBATCH --output=log_TOTAL_%j.out    # Archivo donde verás los "echo" y resultados
#SBATCH --error=log_TOTAL_%j.err     # Archivo para errores
#SBATCH --time=20:00:00              # Duración estimada

# Directorio de salida
mkdir -p resultadosAumentado_PyTorch

# Nombres de los métodos actualizados
METODOS=("Base_Flips" "Geometria" "Oclusion" "Ruido_Sensores" "Firma_Espectral" "Combo_Equilibrado" "All_In")

# 1. INICIALIZACIÓN DE LOGS
# Vaciamos (o creamos) los 7 archivos de log antes de empezar las pruebas.
# Así nos aseguramos de no arrastrar datos de ejecuciones anteriores.
for m_idx in {0..6}
do
    NOMBRE_METODO=${METODOS[$m_idx]}
    > "resultadosAumentado_PyTorch/metodo${NOMBRE_METODO}.log"
done

# 2. EJECUCIÓN DE LOS EXPERIMENTOS
# Iteramos sobre los datasets (0 a 7)
for d_idx in {0..7}
do
    # Iteramos sobre los métodos de aumento (0 a 6)
    for m_idx in {0..6}
    do
        NOMBRE_METODO=${METODOS[$m_idx]}
        ARCHIVO_LOG="resultadosAumentado_PyTorch/metodo${m_idx}_${NOMBRE_METODO}.log"
        
        echo "**Ejecutando dataset: $d_idx | Método: $NOMBRE_METODO | Inicio: $(date +%T)**" | tee -a "$ARCHIVO_LOG"
        
        # Llamada al script: 
        # ID Dataset ($d_idx)
        # Sampler (1 para activarlo)
        # ID Método Aumento ($m_idx)
        python3 -u cnn21_pruebasParalelasBASE_PyTorch.py "$d_idx" 1 "$m_idx" 2>&1 | tee -a "$ARCHIVO_LOG"
        
        echo "FINALIZACIÓN dataset: $d_idx | Método: $NOMBRE_METODO | $(date +%T)" | tee -a "$ARCHIVO_LOG"
        echo "********************************************" | tee -a "$ARCHIVO_LOG"
        
        sleep 3
    done
done

echo "Experimentos finalizados correctamente."