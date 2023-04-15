cmake_minimum_required(VERSION 3.14)

project(gemm)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

include(CPM)

find_package(BLAS)

CPMAddPackage(
  NAME eve
  GITHUB_REPOSITORY jfalcou/eve
  VERSION 2023.02.15
  OPTIONS "EVE_BUILD_TEST OFF" 
	  "EVE_BUILD_BENCHMARKS OFF"
	  "EVE_BUILD_DOCUMENTATION OFF"
)

CPMAddPackage(
  NAME nanobench
  GITHUB_REPOSITORY martinus/nanobench
  VERSION 4.3.11
)

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 9.1.0
)

CPMAddPackage(
  NAME catch2
  GITHUB_REPOSITORY catchorg/Catch2
  VERSION 3.3.2
  OPTIONS "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF"
)

add_library(gemm INTERFACE)
target_include_directories(gemm INTERFACE include)
target_link_libraries(gemm INTERFACE eve::eve)
target_compile_features(gemm INTERFACE cxx_std_20)
add_library(gemm::gemm ALIAS gemm)

# TODO: put in the test directory

add_executable(test test/main.cpp test/nanobench.cpp test/kernels.cpp test/square.cpp)
target_link_libraries(test PRIVATE gemm::gemm nanobench::nanobench fmt::fmt BLAS::BLAS Catch2::Catch2)
target_compile_options(test PRIVATE -march=native)

