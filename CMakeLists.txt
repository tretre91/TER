cmake_minimum_required(VERSION 3.14)

project(eblas)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

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

add_library(eblas INTERFACE)
target_include_directories(eblas INTERFACE include)
target_link_libraries(eblas INTERFACE eve::eve)

add_executable(test test/main.cpp test/nanobench.cpp test/util.cpp)
target_link_libraries(test PRIVATE eblas nanobench::nanobench fmt::fmt BLAS::BLAS)
target_compile_options(test PRIVATE -mavx2 -mfma)

