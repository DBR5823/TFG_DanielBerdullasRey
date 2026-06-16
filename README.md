# Trabajo de Fin de Grado de Daniel Berdullas Rey
## Aumentado de datos basado en máscaras para la clasificación mediante técnicas de aprendizaje profundo de imágenes de teledetección

El uso de sensores multiespectrales de alta resolución facilita la monitorización de ecosistemas complejos como los fluviales con un gran nivel de detalle. Sin embargo, la aplicación de modelos de aprendizaje profundo sobre imágenes multiespectrales capturadas a bordo de un UAV se enfrenta a dos desafíos principales: la escasez de datos etiquetados y el desbalanceo de clases. Estos desafíos suelen provocar problemas de sobreajuste (overfitting), donde los modelos memorizan los datos de entrenamiento en lugar de aprender características generalizables, reduciendo su precisión ante nuevas imágenes.

Este Trabajo de Fin de Grado (TFG) aborda este problema mediante el análisis y la implementación de diversas técnicas de aumentado de datos (Data Augmentation). Se ha diseñado un marco experimental que evalúa métodos tradicionales junto a estrategias de mezcla de muestras (Mixed Sample Data Augmentation), que combinan instancias de entrenamiento mediante procesos probabilísticos de interpolación para regularizar las fronteras de decisión: mixup, CutMix y FMix. Estas estrategias se han probado, combinado y comparado según la métrica de average accuracy sobre las dos arquitecturas de referencia dentro del campo de visión artificial: Redes Neuronales Convolucionales (CNN) y Vision Transformers (ViT), empleando una implementación multi-GPU para la realización de las pruebas.

Los resultados demuestran que el aumentado de datos, combinado con el sobremuestreo de clases minoritarias, es una herramienta esencial para regularizar el entrenamiento en estos escenarios. La estrategia óptima para ViT fue la hibridación de FMix (basado en máscaras fractales) con CutMix (basado en el reemplazo de parches), aplicados a nivel de batch. En el caso de la CNN fue CutMix. En ambos casos se complementaron con técnicas de aumentado clásicas (rotación, zoom y borrado aleatorio de parches), aplicadas a nivel de patch.

Esta configuración consiguió mejorar los resultados de precisión de la clasificación en las clases que son minoritarias en los conjuntos de datos sin repercutir gravemente en la precisión de la clasificación en las clases mayoritarias, mejorando la robustez general del modelo.

Palabras clave / keywords: observación terrestre, aprendizaje profundo, transformers, redes convolucionales, imágenes multiespectrales, aumentado de datos, FMix, CutMix, ecosistemas fluviales.
---

## ⚠️ Consideraciones sobre los Datos de Entrada y Herramientas Propietarias

Por motivos de confidencialidad y derechos de propiedad intelectual pertenecientes a la **Universidade de Santiago de Compostela (USC)**, el directorio `datosEntrada/` (y sus respectivos subdirectorios con archivos `.pgm` y `.raw`) no se incluye en este repositorio público. Estos datos conforman las imágenes de teledetección de alta resolución correspondientes a los ecosistemas fluviales y cuencas hidrográficas de Galicia empleados en este trabajo.

Tampoco se incluyen los segmentos de ejemplo extraídos de los datasets, empleados en la memoria para ejemplificar los diversos métodos de aumentado.

Adicionalmente, para la segmentación y visualización de estas imágenes, se ha hecho uso de **HTOOL**, una herramienta de *software* propiedad de la USC. El trabajo se basa en los ficheros de datos proporcionados por este entorno, los cuales son empleados en los códigos `.py` desarrollados específicamente para los experimentos de este TFG. Por tanto, la reproducción completa de los experimentos requiere disponer de las licencias y accesos pertinentes a estos datos y herramientas.

---

## Estructura Detallada del Repositorio

El directorio de trabajo está diseñado de manera modular, dividiendo los diferentes enfoques experimentales y conservando los ficheros `.log` con la recopilación de los resultados originales:

* **`CodigosBase/Codigos_Imagenes/`**: Contiene los scripts `.py` enfocados en la generación visual de las imágenes asociadas a los métodos de aumentado avanzados empleados en el trabajo (*CutMix*, *mixup* y *FMix*), almacenando ejemplos visuales en directorios de figuras.
* **`Codigos_AumentadoClaseMinoritaria/`**: Directorio dedicado a la **Fase 1** de la experimentación. Incluye los scripts `.py` de entrenamiento y validación de la CNN (junto al fichero `fmix.py` necesario para aplicar *FMix*) empleando técnicas de mezcla avanzadas y sobremuestreo (*oversampling*) de las clases minoritarias. Incluye los ficheros `.sh` para la ejecución automatizada en lote. Los subdirectorios `resultadosPruebas/` y `resultadosPruebasAumentado/` contienen los logs resultantes sin aplicar y aplicando *oversampling*, respectivamente.
* **`Codigos_AumentadosPyTorch/`**: Directorio dedicado a la **Fase 2** y la **Fase 3** de la experimentación. El fichero `cnn21_pruebasParalelasBASE_PyTorch.py` contiene todos los métodos de aumentado clásicos evaluados en la Fase 2, cuyos resultados automatizados mediante `Prueba_Metodos_Base.sh` se almacenan en el subdirectorio `resultadosAumentado_PyTorch/`. El resto de ficheros `.py` y `.sh` ejecutan la experimentación completa de la Fase 3, guardando sus logs en el subdirectorio `resultadosPruebas/`.
* **`Codigos_VIT/`**: Directorio dedicado a la **Fase 4** de la experimentación empleando arquitecturas de *Vision Transformers* (ViT). Agrupa los scripts `.py` de optimización y los ejecutables `.sh` bash para la ejecución automatizada, cuyos resultados se vuelcan en el subdirectorio `resultadosPruevasVIT/`.
* **Análisis y Visualización**: Los *Jupyter Notebooks* (`.ipynb`) procesan los logs generados en cada fase para compilar las métricas finales, organizar las tablas y generar las gráficas comparativas:
  * `Analisis_Fase1.ipynb` (Raíz): Analiza los logs de la Fase 1 obtenidos por `pruebaTOTAL.sh` dentro de `Codigos_AumentadoClaseMinoritaria/`.
  * `Analisis_Fase2.ipynb` (Raíz): Analiza y visualiza los resultados de los métodos de aumento clásicos de la Fase 2 obtenidos mediante `Prueba_Metodos_Base.sh`.
  * `Codigos_AumentadosPyTorch/Analisis_Fase3.ipynb`: Procesa los resultados de las combinaciones complejas de la Fase 3 obtenidas al ejecutar `pruebaTOTAL.sh` en la carpeta de PyTorch.
  * `Codigos_VIT/Analisis_Fase4.ipynb`: Analiza y muestra los resultados de los modelos ViT generados por el script `pruebaTOTAL_VIT.sh`.
  * `comparacionFinal.ipynb` (Raíz): Realiza la comparación definitiva entre las arquitecturas CNN y ViT en sus formas base y sus versiones optimizadas.

---

## Instalación del Software y Dependencias

Para aislar las dependencias y asegurar un entorno reproducible y ligero, el proyecto se ha encapsulado empleando un entorno virtual de `Python` (`venv`). 

### Requisitos Previos e Instalación

1. **Clonar el repositorio y acceder al directorio:**
```bash
git clone https://github.com/DBR5823/TFG_DanielBerdullasRey.git
cd TFG_DanielBerdullasRey

```

2. **Crear y activar el entorno virtual (`venv`):**
* *En Linux/macOS:*
```bash
python3 -m venv tfg-env
source tfg-env/bin/activate

```


* *En Windows:*
```bash
python -m venv tfg-env
tfg-env\Scripts\activate

```




3. **Instalar las dependencias:**
```bash
pip install -r requirements.txt

```



### Archivo `requirements.txt`

```text
torch==2.7.1
torchvision==0.22.1
timm==1.0.25
torchbearer==0.5.5
numpy<2.3.3
pandas==2.3.2
scikit-learn==1.7.2
matplotlib==3.10.8

```

### Compatibilidad de Hardware y Soporte CUDA

El código fuente se adapta automáticamente a las capacidades de cómputo del sistema donde se despliegue:

* **Ejecución en CPU**: Redirige automáticamente la carga a la CPU en equipos sin gráfica dedicada para tareas de depuración, análisis o pruebas con datasets reducidos entre los diversos núcleos.
* **Ejecución en Mono-GPU**: Si el sistema cuenta con soporte CUDA, `PyTorch` aprovechará la aceleración por hardware de forma nativa. Permite realizar experimentos en paralelo empleando varios procesos sobre una única GPU.
* **Ejecución Multi-GPU**: Para entornos de alto rendimiento o servidores, el repositorio incluye scripts optimizados y versiones paralelizadas que distribuyen eficientemente los lotes de datos y los modelos empleando todas las GPUs disponibles.

---

## Manual de Ejecución de los Experimentos

El procedimiento secuencial para replicar los experimentos y generar las métricas de rendimiento es el siguiente:

1. **Configuración inicial**: Active el entorno virtual de `Python` e instale los requerimientos del sistema con `pip install -r requirements.txt`.
2. **Disposición de datos**: Asegúrese de que las imágenes propietarias de la USC, junto a sus datos de segmentación y etiquetas de los segmentos, se encuentren correctamente ubicadas en el directorio `datosEntrada/`. *(Nota: Estos conjuntos de datos no son de acceso libre)*.
3. **Permisos de ejecución**: Es necesario navegar a los directorios de ejecución específicos y habilitar los permisos sobre los scripts bash mediante el comando:
```bash
cd Codigos_AumentadoClaseMinoritaria && chmod +x *.sh && cd ..
cd Codigos_AumentadosPyTorch && chmod +x *.sh && cd ..
cd Codigos_VIT && chmod +x *.sh && cd ..

```

4. **Configuración de rutas del entorno virtual (`venv`)**: Antes de ejecutar los experimentos, se deben modificar los scripts `.sh` para que apunten al entorno virtual de la máquina local. Para ello, abre los scripts de pruebas totales (como `pruebaTOTAL.sh` y `pruebaTOTAL_VIT.sh`) presentes en sus respectivas carpetas, así como el archivo `Codigos_AumentadosPyTorch/Prueba_Metodos_Base.sh`, y edita la línea correspondiente a la activación del entorno virtual sustituyendo la ruta original por la ruta local donde hayas creado tu `venv`.
5. **Lanzamiento de pruebas y Generación de Logs**: Invoca los siguientes scripts desde sus respectivos directorios para generar los archivos `.log`:

    * **Fase 1 (Clase Minoritaria)**: En `Codigos_AumentadoClaseMinoritaria/`, ejecuta el script `pruebaTOTAL.sh` para evaluar de manera secuencial cada tipo de aumentado sobre los 8 datasets.
    * **Fase 2 y 3 (Aumentado Clásico y Combinaciones)**: En `Codigos_AumentadosPyTorch/`, ejecuta `Prueba_Metodos_Base.sh` (para los métodos clásicos de la Fase 2) y posteriormente `pruebaTOTAL.sh` (para las combinaciones complejas de CutMix y CutMix+FMix de la Fase 3).
    * **Fase 4 (Vision Transformers)**: En `Codigos_VIT/`, ejecuta el script `pruebaTOTAL_VIT.sh` para lanzar las pruebas del modelo ViT (base y con aumentos).

6. **Consolidación y Resultados (Jupyter Notebooks)**: Una vez finalizados los entrenamientos, abre y ejecuta los cuadernos interactivos para procesar las métricas y generar las gráficas comparativas definitivas según su ubicación:

    * En la **raíz**: Ejecuta `Analisis_Fase1.ipynb`, `Analisis_Fase2.ipynb` y `comparacionFinal.ipynb`.
    * En **carpetas de origen**: Ejecuta `Codigos_AumentadosPyTorch/Analisis_Fase3.ipynb` y `Codigos_VIT/Analisis_Fase4.ipynb`.



---

## Autor

* [Daniel Berdullas Rey](https://github.com/DBR5823)

*Trabajo desarrollado en el marco de la investigación con la Universidade de Santiago de Compostela (USC).*

