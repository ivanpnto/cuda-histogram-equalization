# Parallel Image Processing & Histogram Equalization

Proyecto desarrollado en C++ y CUDA para el procesamiento de imágenes en paralelo, enfocado en el cálculo de histogramas y algoritmos de ecualización tanto en CPU como en GPU.

## 📂 Estructura del Proyecto

* `CMakeLists.txt`: Archivo de configuración para la compilación con CMake.
* `include/`: Contiene las cabeceras principales del proyecto:
  * `equalization_cpu.h`: Algoritmo de ecualización en CPU.
  * `histogram_cpu.h`: Procesamiento de histogramas secuencial.
  * `histogram_gpu.cuh`: Procesamiento acelerado por GPU utilizando CUDA.
  * `image_io.h`: Utilidades para lectura y escritura de imágenes.
* `external/`: Librerías de terceros (`stb_image` y `stb_image_write`).
* `data/input/`: Imágenes de prueba (ej. `Large.png`).

## ⚙️ Requisitos del Sistema

* Compilador compatible con C++11 o superior.
* [CMake](https://cmake.org/) (versión 3.10 o superior).
* Toolkit de CUDA instalado (para la ejecución en GPU).

## 🚀 Compilación y Ejecución

1. Clona el repositorio y crea una carpeta de compilación:
   ```bash
   git clone [https://github.com/tu-usuario/parallel-image-processing.git](https://github.com/tu-usuario/parallel-image-processing.git)
   cd parallel-image-processing
   mkdir build && cd build
