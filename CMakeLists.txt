cmake_minimum_required(VERSION 3.18)
project(ParalelaProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include external)

add_executable(paralela
    src/main.cpp
    src/image_io.cpp
    src/histogram_cpu.cpp
    src/equalization_cpu.cpp
    src/histogram_gpu.cu
)

set_target_properties(paralela PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)