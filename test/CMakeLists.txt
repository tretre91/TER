find_package(BLAS)

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

set(TEST_SOURCES
  main.cpp
  nanobench.cpp
  kernels.cpp
  square.cpp
  various_dimensions.cpp
  special_cases.cpp
  benchmark.cpp
  benchmark_reporter.cpp
)

add_executable(test ${TEST_SOURCES})
target_link_libraries(test PRIVATE gemm::gemm nanobench::nanobench fmt::fmt BLAS::BLAS Catch2::Catch2)
target_compile_options(test PRIVATE -march=native)

if (GEMM_USE_CTEST)
  list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  include(CTest)
  include(Catch)
  catch_discover_tests(test)
endif()
