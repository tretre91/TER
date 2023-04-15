#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openblas/cblas.h>

#include <gemm/gemm.hpp>

#include "util.hpp"

using Catch::Matchers::WithinRel;

TEST_CASE("Square matrix (n >= 256)") {
	auto dim = GENERATE(256, 512, 1024, 2048);
	// auto dim = GENERATE(16);
	CAPTURE(dim);
	const auto A = util::random_vector<float>(dim * dim);
	const auto B = util::random_vector<float>(dim * dim);
	auto C = util::random_vector<float>(dim * dim);
	auto C2 = C;

	constexpr float alpha = 1.0f;
	constexpr float beta = 1.0f;

	gemm::sgemm(gemm::transposition::none, gemm::transposition::none, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim);
	cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, dim, dim, dim, alpha, A.data(), dim, B.data(), dim,
	  beta, C2.data(), dim);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}
