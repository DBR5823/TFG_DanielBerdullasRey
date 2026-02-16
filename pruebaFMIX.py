import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# 1. IMPORTACIÓN DESDE TU CARPETA LOCAL
# Ajustamos la ruta según tu estructura: implementations/fmix/fmix.py
try:
    from implementations.torchbearer_implementation import FMix
    print("✅ FMix importado correctamente desde 'implementations'")
except ImportError:
    print("❌ Error: No se encontró la carpeta 'implementations/fmix'.")
    print("Asegúrate de que existe el archivo __init__.py en esas carpetas.")
    exit()

def run_fmix_example():
    # 2. CONFIGURACIÓN
    size = (224, 224)
    # alpha: controla la distribución de la mezcla (1.0 es estándar)
    # decay_power: controla la complejidad de la forma de la máscara
    fmix_util = FMix(size=size, alpha=1.0, decay_power=3.0, max_soft=0.0)

    # 3. SIMULACIÓN DE DATOS (BATCH)
    # Creamos un batch de 2 imágenes aleatorias: [Batch, Canales, Alto, Ancho]
    # Usamos valores aleatorios altos y bajos para diferenciar "perro" de "gato"
    img_perro = torch.randn(1, 3, 224, 224) + 2  # Valores altos (más claros)
    img_gato = torch.randn(1, 3, 224, 224) - 2   # Valores bajos (más oscuros)
    batch_x = torch.cat([img_perro, img_gato], dim=0)
    
    # Etiquetas originales (Clase 0 y Clase 1)
    batch_y = torch.tensor([0, 1])

    # 4. APLICAR FMIX
    # Esto genera la máscara de frecuencia y mezcla las imágenes del batch
    mixed_x = fmix_util(batch_x)
    
    # Datos de la mezcla necesarios para la función de pérdida (Loss)
    indices = fmix_util.index  # Índices de las imágenes con las que se mezcló
    lam = fmix_util.lam        # Coeficiente de mezcla (Lambda)

    print(f"--- Info de la Mezcla ---")
    print(f"Lambda (peso imagen principal): {lam:.4f}")
    print(f"Índices de mezcla: {indices}")
    print(f"La imagen 0 ahora es {lam*100:.1f}% Clase A y {(1-lam)*100:.1f}% Clase B")

    # 5. PREPARACIÓN PARA VISUALIZACIÓN
    # Función para normalizar el tensor al rango [0, 1] y evitar el error de clipping
    def prepare_for_plot(tensor):
        t = tensor.clone().detach().cpu()
        t = (t - t.min()) / (t.max() - t.min()) # Reescalado min-max
        return t.permute(1, 2, 0).numpy()       # De [C, H, W] a [H, W, C]

    # 6. GUARDAR RESULTADO
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Imagen mezclada 1
    axs[0].imshow(prepare_for_plot(mixed_x[0]))
    axs[0].set_title(f"Mezcla 1 (lam: {lam:.2f})")
    axs[0].axis('off')
    
    # Imagen mezclada 2
    axs[1].imshow(prepare_for_plot(mixed_x[1]))
    axs[1].set_title(f"Mezcla 2")
    axs[1].axis('off')

    output_file = "resultado_fmix_tfg.png"
    plt.tight_layout()
    plt.savefig(output_file)
    
    print(f"--- Finalizado ---")
    print(f"✅ Visualización guardada en: {output_file}")

if __name__ == "__main__":
    run_fmix_example()
