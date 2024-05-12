# AMD Zen HPL

Derived from:
>  High Performance Linpack Benchmark (HPL)
>  HPL - 2.3 - December 2, 2018

## Description:

HPL is a software package that solves a (random) dense linear
system  in   double  precision  (64   bits)   arithmetic   on 
distributed-memory  computers.   It can thus be regarded as a
portable as well as  freely  available implementation  of the
High Performance Linpack Benchmark.

This version has been optimized to run on AMD CPUs.

-----------------------------------------------------------------------------

## RELEASE 2023_07

### Dependencies:

* OpenMPI 4.x:  This release was built with OpenMPI 4.1.5, and should run
  without issue with any build of OpenMPI 4.x
* This release was built on Rocky Linux 8.6, and tested on CentOS 8.3, Rocky
  Linux 9.0, and Ubuntu Linux 22.04.
* This release includes changes from 2023_05 to address Top500 compatibility.

### Recommended system settings:

* BOOST: ON
* SMT: OFF
* NPS: 4
* Determinism: Power
* Transparent Hugepages: always


### How to run:

1. Ensure the above dependencies have been satisfied, and `mpirun` from
   OpenMPI 4.x is on your `$PATH`
2. Optionally, run the `./reset-system.sh` script to set various OS parameters
   to recommended values.
3. Optionally, create an HPL.dat file, as described in the documentation.
   AMD Zen HPL will auto-tune to select reasonable default parameters, so an
   HPL.dat file is not required.
   By default, AMD Zen HPL will attempt to use roughly 90% of system memory.
4. For single-node runs, run `./run.sh`.  For multi-node runs, adapt the
   `mpirun` line from `run.sh` to suit the local site.
   
   It is important to set process bindings, and set the `OMP_NUM_THREADS`
   variable to the proper number of threads to be used per rank.

   Recommended arguments to `mpirun` are:

   * `--map-by socket:PE=$NUM_CORES_PER_SOCKET` - to specify 1 MPI rank per
     socket, with $NUM_CORES_PER_SOCKET (ideally set to number of cores per
     socket) CPUs assigned to the rank.
   * `-x OMP_NUM_THREADS=$NUM_CORES_PER_SOCKET` - to inform OpenMP of how many
     threads to use.
   * `-x OMP_PROC_BIND=spread -x OMP_PLACES=cores` - to guide OpenMP how to
     distribute threads.

-----------------------------------------------------------------------------
