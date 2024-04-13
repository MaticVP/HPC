cmake_minimum_required(VERSION 3.26)
project(DN2 CUDA CXX C)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_FLAGS -fopenmp)
set(CMAKE_CXX_FLAGS -fopenmp)

find_package(CUDAToolkit REQUIRED)

set(CPP_SOURCES
        main.cpp
        histogram_norm_seq.c
)

set(CUDA_SOURCES
        historgram_norm_cuda.cu
        cumulative_histogram.cu
        compute_histogram.cu
)

add_executable(DN2 ${CPP_SOURCES} ${CUDA_SOURCES})

set_target_properties(DN2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(DN2 CUDA::cudart)
