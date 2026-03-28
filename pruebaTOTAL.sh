#!/bin/bash
#SBATCH --job-name=EXP_TOTAL_DANIEL
#SBATCH -w ctgpgpu5                  # Fuerza el nodo actual
#SBATCH --gres=gpu:2                 # Reserva las 2 GPUs
#SBATCH --cpus-per-task=7            # Reserva exactamente 6 núcleos de CPU
#SBATCH --partition=gpu              # Partición de GPUs del CITIUS
#SBATCH --output=log_TOTAL_%j.out    # Archivo donde verás los "echo" y resultados
#SBATCH --error=log_TOTAL_%j.err     # Archivo para errores
#SBATCH --time=20:00:00              #Duración estimada


source /home/daniel.berdullas/TFG/venv/bin/activate


echo "INICIO EXPERIMENTACIÓN COMPLETA: $(date +%T)"
echo "Nodo de ejecución: $SLURM_NODELIST"
echo "GPUs asignadas: $CUDA_VISIBLE_DEVICES"

echo "----------------------------------------"


echo "Iniciando pruebas CNN_Basica..."
./pruebaCNN_Basica.sh

sleep 1

echo "Iniciando pruebas FMIX..."
./pruebaFMIX.sh

sleep 1

echo "Iniciando pruebas CUTMIX..."
./pruebaCUTMIX.sh

sleep 1

echo "Iniciando pruebas MIXUP..."
./pruebaMIXUP.sh

sleep 1

echo "Iniciando pruebas CUTMIX+FMIX..."
./pruebaCUTMIX_FMIX.sh

sleep 1

rm *.json

echo "----------------------------------------"
echo "FIN EXPERIMENTACIÓN COMPLETA: $(date +%T)"
