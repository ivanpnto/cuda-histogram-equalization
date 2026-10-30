cmake_minimum_required(VERSION 3.18)

# GPU architecture. Defaults to Turing (sm_75, e.g. the NVIDIA T4 used for the
# benchmarks). Override with -DCMAKE_CUDA_ARCHITECTURES=<arch>, e.g. 86 or 89.
# Must be set before project(), which otherwise picks the compiler default.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(cuda_histogram_equalization LANGUAGES CXX CUDA)

# Default to an optimized build. Without it, single-config generators compile
# the CPU baseline without optimizations and the reported speedups are inflated.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(histogram_equalization
    src/main.cpp
    src/image_io.cpp
    src/histogram_cpu.cpp
    src/equalization_cpu.cpp
    src/histogram_gpu.cu
)

target_include_directories(histogram_equalization PRIVATE include external)

set_target_properties(histogram_equalization PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
