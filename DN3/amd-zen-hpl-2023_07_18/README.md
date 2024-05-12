# AMD Zen HPL 2023_07_18

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

## New Features:

* Auto-detection of reasonable starting points:
If no HPL.dat file is found in the current working directory,
ADM Zen HPL will attempt to pick reasonable parameters.  There
is no guarantee that these parameters will produce the best
results, but may be used as a guide to start a benchmarking
exercise.

* Saving of auto-detection values is accomplished via 'create'
mode.  Pass the '-c' or '--create' argument will cause HPL
to generate a new HPL.dat file with auto-detected values.
Pass the '-f <filename>' flag to specify an output filename
other than 'HPL.dat'

* Specifying HPL.dat file:
The '-f <datfile>' commandline parameter may be used to specify
what HPL.dat file to use.  Defaults to 'HPL.dat', but this
allows overriding to specify a different input file.

* Stopping Early:
The '-s <stopcol>' stops HPL processing after <stopcol> number of
columns have been processed.  '-S <startcol>' begins processing at
startcol, rather than column 0.  Results are not valid, but this can
be useful in running parameter sweeps of larger matrix sizes, or testing
how HPL behaves in different phases of processing.

* Overriding test parameters:
The '-o <override>' commandline parameter allows the user to
override the parameters that would have been used.  This flag
may be used multiple times.

  xhpl -o N:1000,2000 -o PQ:2x4,4x2

See the argument help for more information (xhpl --help)

* Runtime flags to display progress and print detailed timing info
  * Use '-p' to show progress.
  * Use '-t' to show detailed timing info, use twice to see per-rank details.
  * Use '-P' to write progress in CSV format to a log file.

* Displaying Effective CPU Frequency:
When printing progress status, also print the average effective CPU
frequency of Rank 0 processor.

* Add customizations to Spread-Roll (long) swap algorithm (1)
You can specify more than one number on on the SWAP line to test
multiple configurations.  For each configuration, you can tune the
selection of the Spread and Roll algorithms used:
  * Specify '0' in 10's position for original spread algorithm
  * Specify '1' in 10's position for spread via collectives
  * Specify '2' in 10's position for a direct spread
  * Specify '0' in 100's position for original roll algorithm
  * Specify '1' in 100's position for roll via allgatherv
  * Specify '2' in 100's position for roll via broadcasts

* Add customization to Panel Factoring MXSwap. Add new line 32 to 
of HPL.dat to specify '0' for the original algorithm, or '1' to 
choose a MPI collectives-based algorithm.  Multiple values
can be specified on line 32 to test both algorithms

* Experimental support for Row-Major layout of A Matrix:
HPL will transpose the matrix automatically if ATRANS is '1'.  The transpose
time is included as part of the solve time, and thus counts against the total
performance. It will most likely not be beneficial at small scale, but at very
large scales, the time taken to transpose the matrix may be overcome by
optimizations in the row swapping.

Other changes:

* Removed support for U to be stored in ColumnMajor format. There
are no cases where this format is better, and it allows for
simplifying the code.

* Build-time support for reading raw performance counters around monitored
areas of code (such as the Update DGEMM) with `-DHPL_PERFORMANCE_MONITOR`
