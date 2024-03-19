cmake_minimum_required(VERSION 3.26)
project(DN1 C)
project(DN1_Basic_Parallel C)

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_FLAGS -fopenmp)
#add_executable(DN1 basic_seq.c)
#add_executable(DN1 basic_parallel.c)
add_executable(DN1 optimize_seq.c)
