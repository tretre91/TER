#ifndef GEMM_TEST_UTIL_HPP
#define GEMM_TEST_UTIL_HPP

#include <catch2/catch_config.hpp>
#include <catch2/catch_get_random_seed.hpp>
#include <nanobench.h>
#include <openblas/cblas.h>

#include <algorithm>
#include <random>
#include <vector>

#include <gemm/gemm.hpp>

namespace util
{
	// Precision for floating point relative comparisons
	template<typename T>
	constexpr T precision = 1e-5;

	template<>
	inline constexpr auto precision<float> = 1e-3f;

	template<>
	inline constexpr auto precision<double> = 1e-7;

	// transposition setting
	constexpr auto no_trans = gemm::transposition::none;

	// Current Catch2 session config data
	inline Catch::ConfigData config_data;

	// Benchmark runner
	inline auto bench = ankerl::nanobench::Bench().warmup(10).relative(true);

	// Returns a random floating point value
	template<typename T>
	T random_float() {
		static std::mt19937 gen(Catch::getSeed());
		static std::uniform_real_distribution<T> dist(-10, 10);
		return dist(gen);
	}

	// Generates a vector of random floating point values
	template<typename T>
	std::vector<T> random_vector(int size) {
		std::vector<T> v(size);
		for (auto& e : v) {
			e = random_float<T>();
		}

		return v;
	}

	// Wrapper for the gemm function
	template<typename T>
	void gemm(const int M, const int N, const int K, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc) {
		gemm::gemm<T>(gemm::transposition::none, gemm::transposition::none, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}

	// Wrapper for the openblas [s|d]gemm function
	template<typename T>
	void cblas_gemm(
	  const int M, const int N, const int K, const T alpha, const T* A, const int lda, const T* B, const int ldb, const T beta, T* C, const int ldc) {
		if constexpr (std::is_same_v<T, float>) {
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
		} else {
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
		}
	}

	// Naive matrix multiplication
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
