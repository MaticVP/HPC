cmake_minimum_required(VERSION 3.26)
project(DN1 C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS -fopenmp)
add_executable(DN1 sample.c)
