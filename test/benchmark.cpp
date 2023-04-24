#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <nanobench.h>
#include <openblas/cblas.h>

#include "util.hpp"
#include <gemm/gemm.hpp>

TEST_CASE("M,N,K <= 8", "[.benchmark][small]") {
	using util::bench;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	const auto M = GENERATE(1, 2, 4, 8);
	const auto K = GENERATE(1, 2, 4, 8);

	const auto A = util::random_vector<float>(M * K);

	const auto N = GENERATE(1, 2, 4, 8);

	CAPTURE(M, N, K);
	DYNAMIC_SECTION("" << M << "x" << K << " * " << K << "x" << N) {
		const auto B = util::random_vector<float>(K * N);
		auto C = util::random_vector<float>(M * N);
		auto oldC = C;

		const auto* ptr_A = A.data();
		const auto* ptr_B = B.data();
		auto* ptr_C = C.data();

		bench.title(fmt::format("{}x{} * {}x{}", M, K, K, N));
		bench.run("gemm", [=] {
			gemm::sgemm(gemm::transposition::none, gemm::transposition::none, M, N, K, alpha, ptr_A, K, ptr_B, N, beta, ptr_C, N);
		});
		std::copy(oldC.begin(), oldC.end(), C.begin());

		bench.run("naive", [=] { util::naive_gemm(M, N, K, alpha, ptr_A, K, ptr_B, N, beta, ptr_C, N); });
		std::copy(oldC.begin(), oldC.end(), C.begin());

		bench.run("blas", [=] {
			cblas_sgemm(
			  CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, alpha, ptr_A, K, ptr_B, N, beta, ptr_C, N);
		});
	}
}

TEST_CASE("M,N,K >= 64", "[.benchmark][large]") {
	using util::bench;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	bench.warmup(0);

	const auto M = GENERATE(64, 128, 256, 512);
	const auto K = GENERATE(64, 128, 256, 512);

	const auto A = util::random_vector<float>(M * K);

	const auto N = GENERATE(64, 128, 256, 512);

	CAPTURE(M, N, K);
	DYNAMIC_SECTION("" << M << "x" << K << " * " << K << "x" << N) {
		const auto B = util::random_vector<float>(K * N);
		auto C = util::random_vector<float>(M * N);
		auto oldC = C;

		const auto* ptr_A = A.data();
		const auto* ptr_B = B.data();
		auto* ptr_C = C.data();

		bench.title(fmt::format("{}x{} * {}x{}", M, K, K, N));
		bench.run("gemm", [=] { gemm::sgemm(gemm::transposition::none, gemm::transposition::none, M, N, K, alpha, ptr_A, K, ptr_B, N, beta, ptr_C, N); });
		std::copy(oldC.begin(), oldC.end(), C.begin());

		if (M < 512 && K < 512 && N < 512) {
			bench.run("naive", [=] { util::naive_gemm(M, N, K, alpha, ptr_A, K, ptr_B, N, beta, ptr_C, N); });
			std::copy(oldC.begin(), oldC.end(), C.begin());
		}

		bench.run("blas", [=] {
			cblas_sgemm(
			  CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, alpha, ptr_A, K, ptr_B, N, beta, ptr_C, N);
		});
	}
}

TEST_CASE(">= 1024", "[.benchmark][very large]") {
	using util::bench;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	bench.warmup(0);

	const auto dim = GENERATE(1024, 2000, 4096, 8000, 8192);

	CAPTURE(dim);
	DYNAMIC_SECTION("" << dim << "x" << dim << " * " << dim << "x" << dim) {
		const auto A = util::random_vector<float>(dim * dim);
		const auto B = util::random_vector<float>(dim * dim);
		auto C = util::random_vector<float>(dim * dim);
		auto oldC = C;

		const auto* ptr_A = A.data();
		const auto* ptr_B = B.data();
		auto* ptr_C = C.data();

		bench.title(fmt::format("{0}x{0}", dim));
		bench.run(
		  "gemm", [=] { gemm::sgemm(gemm::transposition::none, gemm::transposition::none, dim, dim, dim, alpha, ptr_A, dim, ptr_B, dim, beta, ptr_C, dim); });
		std::copy(oldC.begin(), oldC.end(), C.begin());

		bench.run("blas", [=] {
			cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, dim, dim, dim, alpha, ptr_A, dim, ptr_B, dim,
			  beta, ptr_C, dim);
		});
	}
}
