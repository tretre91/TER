# gemm

A C++ implementation of blas matrix-matrix multiplication routines (xgemm).

## Build

The library is header only, you just have to copy the files located in the `include` folder to use it.

In order to build the tests/benchmarks, you will have to use [CMake](https://cmake.org/) or [xmake](https://xmake.io),
and have openblas installed on your system.

## TODO

- Integrate the kernels to the main matrix multiplication function
- Microkernels
	- [x] Kernel composition function
	- [x] 1x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
	- [x] 2x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
	- [ ] 4x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
	- [ ] 8x(1, 2, 4, 8)x(1, 2, 4, 8) kernels
- Tests
	- [x] Kernels
	- [ ] Big matrices
	- [ ] Fix precision issues
- Benchmarks
	- [ ] Kernels
	- [ ] Big matrices

