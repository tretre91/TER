#ifndef GEMM_TEST_UTIL_HPP
#define GEMM_TEST_UTIL_HPP

#include <catch2/catch_get_random_seed.hpp>
#include <openblas/cblas.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace util
{
	// Precision for floating point relative comparisons
	template<typename T>
	constexpr T precision = 1e-5;

	// ouput stream for the benchmarks
	inline std::ostream* benchmark_output = &std::cout;

	// Generates a vector of random floating point values
	template<typename T>
	std::vector<T> random_vector(int size) {
		static std::mt19937 gen(Catch::getSeed());
		static std::uniform_real_distribution<T> dist(-10, 10);

		std::vector<float> v(size);
		for (auto& e : v) {
			e = dist(gen);
		}

		return v;
	}

	/**
	 * @brief Computes a simple matrix multiplication with openblas
	 * @note the arrays passed as parameters are expected to be contiguous
	 */
	inline void openblas_sgemm(
	  int M, int N, int K, float alpha, std::span<const float> A, int lda, std::span<const float> B, int ldb, float beta, std::span<float> C, int ldc) {
		cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, alpha, A.data(), lda, B.data(), ldb,
		  beta, C.data(), ldc);
	}

	template<typename T>
	void naive_gemm(int M, int N, int K, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc) {
		for (int i = 0; i < M; i++) {
			std::transform(&C[i * ldc], &C[(i + 1) * ldc], &C[i * ldc], [&](auto x) { return beta * x; });
			for (int k = 0; k < K; k++) {
				for (int j = 0; j < N; j++) {
					C[j + i * ldc] += alpha * A[k + i * lda] * B[j + k * ldb];
				}
			}
		}
	}
} // namespace util

#endif
