#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "util.hpp"
#include <gemm/detail/kernels.hpp>
#include <openblas/cblas.h>

using Catch::Matchers::WithinRel;

TEST_CASE("Microkernels", "[kernels][small]") {
	Catch::StringMaker<float>::precision = 15;

	constexpr float alpha = 1.0f;
	constexpr float beta = 1.0f;

	const auto M = GENERATE(range(1, 9));
	const auto K = GENERATE(range(1, 9));

	const auto A = util::random_vector<float>(M * K);

	const auto N = GENERATE(range(1, 9));

	CAPTURE(M, N, K);

	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	auto C2 = C;

	gemm::detail::get_kernel<float>(M, N, K)(A.data(), K, B.data(), N, C.data(), N);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C2.data(), N);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

