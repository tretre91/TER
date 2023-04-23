#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openblas/cblas.h>

#include <gemm/gemm.hpp>

#include "util.hpp"

using Catch::Matchers::WithinRel;

TEST_CASE("various dimensions", "[rectangle]") {
	auto M = GENERATE(10, 72, 256, 329);
	auto N = GENERATE(8, 64, 257);
	auto K = GENERATE(1, 10, 100);
	
	CAPTURE(M, N, K);
	const auto A = util::random_vector<float>(M * K);
	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	auto C2 = C;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	gemm::sgemm(gemm::transposition::none, gemm::transposition::none, M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N);
	cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C2.data(), N);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

