# Trabajo de Fin de Grado de Daniel Berdullas Rey
## Optimización de CNNs y Vision Transformers mediante Técnicas de Regularización y Aumento de Datos Espaciales

El uso de sensores multiespectrales de alta resolución facilita la monitorización de ecosistemas complejos como los fluviales con un gran nivel de detalle. Sin embargo, la aplicación de modelos de aprendizaje profundo en este ámbito se enfrenta a un desafío principal: la escasez de datos etiquetados a nivel de segmento. Esta falta de muestras etiquetadas suele provocar problemas de sobreajuste (*overfitting*), donde los modelos memorizan los datos de entrenamiento en lugar de aprender características generalizables, reduciendo su precisión ante nuevas imágenes.

Este Trabajo de Fin de Grado (TFG) aborda este problema mediante el análisis y la implementación de diversas técnicas de aumentado de datos (*Data Augmentation*). Se ha diseñado un marco experimental que evalúa métodos tradicionales frente a algoritmos avanzados: **MixUp**, **CutMix** y **FMix**. Estas estrategias se han probado y comparado sobre las dos arquitecturas de referencia dentro del campo de visión artificial: Redes Neuronales Convolucionales (CNN) y *Vision Transformers* (ViT).

Los resultados demuestran que el aumentado de datos, combinado con el *oversampling* de clases minoritarias, es una herramienta esencial para regularizar el entrenamiento en estos escenarios. En concreto, la hibridación de **FMix** (basado en máscaras fractales) con **CutMix** (basado en el reemplazo de parches), aplicados a nivel de *batch*, sobre técnicas de aumentado clásicas (rotación, zoom y borrado aleatorio de parches), aplicadas a nivel de *patch*, se consolidó como la estrategia óptima. Esta combinación logró el mejor rendimiento de clasificación en ambas redes y consiguió equilibrar el aprendizaje en las clases minoritarias del ecosistema, mejorando la robustez general del modelo.

**Palabras clave / keywords:** teledetección, aprendizaje profundo, imágenes multiespectrales, aumentado de datos, *FMix*, *CutMix*, ecosistemas fluviales.

---

## ⚠️ Consideraciones sobre los Datos de Entrada y Herramientas Propietarias

Por motivos de confidencialidad y derechos de propiedad intelectual pertenecientes a la **Universidade de Santiago de Compostela (USC)**, el directorio `datosEntrada/` (y sus respectivos subdirectorios con archivos `.pgm` y `.raw`) no se incluye en este repositorio público. Estos datos conforman las imágenes de teledetección de alta resolución correspondientes a diferentes ecosistemas fluviales de Galicia.

Adicionalmente, para la segmentación y visualización de estas imágenes, se ha hecho uso de **HTOOL**, una herramienta de *software* propiedad de la USC. El trabajo se basa en los ficheros de datos proporcionados por este entorno, los cuales son empleados en los códigos `.py` desarrollados específicamente para los experimentos de este TFG. Por tanto, la reproducción completa de los experimentos requiere disponer de las licencias y accesos pertinentes a estos datos y herramientas.

---

## 📂 Estructura Detallada del Repositorio

El directorio de trabajo está diseñado de manera modular, dividiendo los diferentes enfoques experimentales y la recopilación de resultados:

* **`CodigosBase/`**: Contiene los scripts `.py` elementales enfocados en la generación visual y teórica de las imágenes tratadas mediante las técnicas de aumentado espacial (*CutMix*, *MixUp* y *FMix*), almacenando ejemplos visuales en directorios de figuras.
* **`Codigos_AumentadosPyTorch/`**: Núcleo de las implementaciones puras en PyTorch. Agrupa los scripts `.py` que definen los modelos y los procesos de entrenamiento con diferentes aumentos de datos. Asimismo, incluye múltiples archivos `.sh` (ej. `pruebaTOTAL.sh`, `pruebaCUTMIX.sh`) que permiten la ejecución automatizada en lotes, volcando la salida estándar en archivos `.log` dentro del subdirectorio `resultadosAumentado_PyTorch/`.
* **`Codigos_AumentadoClaseMinoritaria/`**: Directorio dedicado a la mitigación del desbalanceo de clases. Incluye los códigos desarrollados para aplicar técnicas de mezcla y aumentado exclusivamente sobre las clases minoritarias. Destaca el uso de versiones paralelizadas (como `cnn_21_CUTMIX_FMIX_PARALELO.py`) para optimizar drásticamente los tiempos computacionales.
* **`Codigos_VIT/`**: Contenedor de las pruebas experimentales empleando arquitecturas de *Vision Transformers* (ViT). Sigue un patrón similar al resto, aportando sus propios scripts paralelos y ejecutables bash para obtener los correspondientes `.log` de resultados bajo estrés espacial y mezclas probabilísticas.
* **Análisis y Visualización (Raíz)**: En el directorio raíz se exponen los *Jupyter Notebooks* (`.ipynb`) y sus exportaciones a HTML que procesan los archivos de *logs* generados en los pasos anteriores para compilar las métricas finales y organizar las tablas y gráficas comparativas:
  * `creacionTablas.ipynb`: Analiza los `.log` obtenidos por `pruebaTOTAL.sh` de la carpeta `Codigos_AumentadoClaseMinoritaria`.
  * `ComparacionMetodosPytorch.ipynb`: Analiza y visualiza los resultados de los métodos de aumento clásicos obtenidos mediante `Prueba_Metodos_Base.sh`.
  * `ComparacionMetodosPyTorch_CUTMIX+FMIX.ipynb`: Procesa los resultados de las combinaciones complejas obtenidas al ejecutar `pruebaTOTAL.sh` en la carpeta de PyTorch.
  * `comparacionMetodosVIT.ipynb`: Analiza y muestra los resultados generados por el script `pruebaTOTAL_VIT.sh`.
  * `comparacionFinal.ipynb`: Realiza la comparación definitiva entre las arquitecturas CNN y ViT en sus formas base y sus versiones optimizadas.

---

## Instalación del Software y Dependencias

Para aislar las dependencias y asegurar un entorno reproducible y ligero, el proyecto se ha encapsulado empleando un entorno virtual de `Python` (`venv`). 

### Requisitos Previos e Instalación

1. **Clonar el repositorio y acceder al directorio:**
   ```bash
   git clone [https://github.com/DBR5823/TFG_AumentadoDatos.git](https://github.com/DBR5823/TFG_AumentadoDatos.git)
   cd TFG_AumentadoDatos

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
Tal y como se especifica en el archivo `requirements.txt`, la configuración se enfoca íntegramente en el ecosistema de `PyTorch` y el análisis clásico de datos:
```bash
pip install -r requirements.txt

```



### Archivo `requirements.txt`

```text
torch==2.7.1
torchvision==0.22.1
timm==1.0.25
torchbearer==0.5.5
numpy==2.3.3
pandas==2.3.2
scikit-learn==1.7.2
matplotlib==3.10.8

```

### Compatibilidad de Hardware y Soporte CUDA

El código fuente ha sido diseñado de manera flexible para adaptarse automáticamente a las capacidades de cómputo del sistema donde se despliegue:

* **Ejecución en CPU**: Redirige la carga a la CPU en equipos sin gráfica dedicada para tareas de depuración o pruebas con datasets reducidos.
* **Ejecución en Mono-GPU**: Si el sistema cuenta con soporte CUDA, `PyTorch` aprovechará la aceleración por hardware de forma nativa para agilizar significativamente el entrenamiento y la inferencia.
* **Ejecución Multi-GPU**: Para entornos de alto rendimiento o servidores, el repositorio incluye scripts optimizados que distribuyen eficientemente los lotes de datos y los modelos entre todas las GPUs disponibles.

---

## Manual de Ejecución de los Experimentos

El procedimiento secuencial para replicar los experimentos y generar las métricas de rendimiento para las cuencas fluviales es el siguiente:

1. **Configuración inicial**: Active el entorno virtual de `Python` e instale las dependencias.
2. **Disposición de datos**: Asegúrese de que las imágenes propietarias de la USC y la herramienta **HTOOL** se encuentren mapeadas y ubicadas en el directorio `datosEntrada/`.
3. **Permisos de ejecución**: Habilite los permisos sobre los scripts bash mediante los siguientes comandos:
```bash
chmod +x Codigos_AumentadoClaseMinoritaria/*.sh
chmod +x Codigos_AumentadosPyTorch/*.sh
chmod +x Codigos_VIT/*.sh

```


4. **Lanzamiento de pruebas y Generación de Logs**:
Invoque los siguientes scripts desde sus respectivos directorios:
* **Aumentado en Clase Minoritaria**: Ejecute `pruebaTOTAL.sh` dentro de `Codigos_AumentadoClaseMinoritaria/`.
* **Aumentado Clásico y Combinaciones**: En `Codigos_AumentadosPyTorch/`, ejecute `Prueba_Metodos_Base.sh` (métodos clásicos) y posteriormente `pruebaTOTAL.sh` (combinación con CutMix/FMix).
* **Vision Transformers (ViT)**: Ejecute `pruebaTOTAL_VIT.sh` dentro de la carpeta `Codigos_VIT/`.


5. **Consolidación y Resultados (Jupyter Notebooks)**: Una vez finalizado el proceso, ejecute los *Jupyter Notebooks* interactivos de la raíz (`creacionTablas.ipynb`, `ComparacionMetodosPytorch.ipynb`, `ComparacionMetodosPyTorch_CUTMIX+FMIX.ipynb`, `comparacionMetodosVIT.ipynb` y `comparacionFinal.ipynb`) para interpretar visualmente la evolución del *loss* y generar las gráficas comparativas de exactitud por clase.

---

## Autor

* [Daniel Berdullas Rey](https://github.com/DBR5823)

*Trabajo desarrollado en el marco de la investigación con la Universidade de Santiago de Compostela (USC).*
