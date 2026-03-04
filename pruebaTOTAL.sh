#!/bin/bash


echo "INICIO EXPERIMENTACIÓN COMPLETA: $(date +%T)"

echo "Iniciando pruebas CNN_Basica..."
./pruebaCNN_Basica.sh

echo "Iniciando pruebas FMIX..."
./pruebaFMIX.sh


echo "Iniciando pruebas CUTMIX..."
./pruebaCUTMIX.sh


echo "Iniciando pruebas MIXUP..."
./pruebaMIXUP.sh


echo "FIN EXPERIMENTACIÓN COMPLETA: $(date +%T)"
