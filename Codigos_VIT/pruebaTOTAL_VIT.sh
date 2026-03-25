#!/bin/bash
#SBATCH --job-name=EXP_TOTAL_DANIEL
#SBATCH -w ctgpgpu5                  # Fuerza el nodo actual
#SBATCH --gres=gpu:2                 # Reserva las 2 GPUs
#SBATCH --partition=gpu              # Partición de GPUs del CITIUS
#SBATCH --output=log_TOTAL_%j.out    # Archivo donde verás los "echo" y resultados
#SBATCH --error=log_TOTAL_%j.err     # Archivo para errores
#SBATCH --time=100:00:00              #Duración estimada


source /home/daniel.berdullas/TFG/venv/bin/activate


echo "INICIO EXPERIMENTACIÓN COMPLETA: $(date +%T)"
echo "Nodo de ejecución: $SLURM_NODELIST"
echo "GPUs asignadas: $CUDA_VISIBLE_DEVICES"

echo "----------------------------------------"


echo "Iniciando pruebas VIT_Basica..."
#./pruebaVIT_Basica.sh

sleep 1



echo "Iniciando pruebas CUTMIX..."
#./pruebaVIT_CUTMIX.sh

sleep 1

echo "Iniciando pruebas CUTMIX+FMIX..."
./pruebaVIT_CUTMIX_FMIX.sh

sleep 1

rm *.json

echo "----------------------------------------"
echo "FIN EXPERIMENTACIÓN COMPLETA: $(date +%T)"
