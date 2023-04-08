# eblas

A C++ implementation of blas matrix-matrix multiplication routines (xgemm).

## TODO

- Microkernels
- Real tests

## Build

The library is header only, you just have to copy the files located in the `include` folder to use it.

In order to build the tests/benchmarks, you will have to use [CMake](https://cmake.org/) or [xmake](https://xmake.io),
and have openblas installed on your system.

