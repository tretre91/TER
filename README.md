# gemm

A C++ implementation of blas matrix-matrix multiplication routines (xgemm).

## Build

The library is header only, you just have to copy the files located in the
`include` folder to use it.

In order to build the tests/benchmarks, you will have to use
[CMake](https://cmake.org/) or [xmake](https://xmake.io), and have openblas
installed on your system.

## Benchmark

Some benchmark results are available [here](./misc/benchmark.md), they were ran
on a AMD Ryzen 5 2600 cpu, locked at 3400MHz. The program was compiled with
`g++` and the `-O3 -march=native` options.

The floating point version `gemm<float>` was run against openblas' `sgemm` and,
for small enough matrices, a naive algorithm (with the loop swapping
optimization).

## TODO

- Integrate the kernels to the main matrix multiplication function
	- [x] Have a working implementation
	- [ ] Fix performance issues
- Microkernels
	- [x] Kernel composition function
	- [x] 1x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
	- [x] 2x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
	- [x] 4x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
	- [x] 8x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
- Tests
	- [x] Kernels
	- [x] Big matrices
	- [ ] Fix precision issues
	- [x] Write a custom Catch2 reporter
	- [x] Double
- Benchmarks
	- [x] Small matrices
	- [x] Big matrices
	- [ ] Double

