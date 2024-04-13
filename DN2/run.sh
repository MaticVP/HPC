#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --job-name=code_sample
#SBATCH --gpus=1
#SBATCH --output=sample_out.log

module load CUDA
module load CMake

mkdir build
cd build

cmake ..
cmake --build .

srun  ./DN2 ../valve.png ../valve_out.png