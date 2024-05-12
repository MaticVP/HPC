#!/bin/bash
#SBATCH --reservation=fri
#SBATCH --job-name=hpl-benchmark
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=hpl_benchmark_tune.log

module load OpenMPI/4.1.5-GCC-12.3.0

unset OMPI_MCA_osc

NT=$(lscpu | awk '/per socket:/{print $4}')
NR=2
MAP_BY=socket

MAP_BY=socket
NUM_CORES_PER_SOCKET=64

export UCX_TLS=self, tcp
mpirun --map-by socket:PE=$NUM_CORES_PER_SOCKET  -x OMP_NUM_THREADS=128 -x OMP_PROC_BIND=spread -x OMP_PLACES=cores -np $SLURM_NTASKS ./xhpl -p