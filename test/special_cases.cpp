#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openblas/cblas.h>

#include <gemm/gemm.hpp>

#include "util.hpp"

using Catch::Matchers::WithinRel;

TEST_CASE("worst case", "[square][large][special]") {
	constexpr auto dim = gemm::detail::B2 + gemm::detail::B1 + gemm::detail::BT<float> + 1;

	const auto A = util::random_vector<float>(dim * dim);
	const auto B = util::random_vector<float>(dim * dim);
	auto C = util::random_vector<float>(dim * dim);
	auto C2 = C;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	gemm::sgemm(gemm::transposition::none, gemm::transposition::none, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim);
	cblas_sgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, dim, dim, dim, alpha, A.data(), dim, B.data(), dim,
	  beta, C2.data(), dim);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

TEST_CASE("dot product", "[large][special]") {
	constexpr int M = 1;
	constexpr int N = 1;
	constexpr int K = gemm::detail::B2 + gemm::detail::B1 + gemm::detail::BT<float> + 1;

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

TEST_CASE("outer product", "[large][special]") {
	constexpr int M = gemm::detail::B2 + gemm::detail::B1 + gemm::detail::BT<float> + 1;;
	constexpr int N = gemm::detail::B2 + gemm::detail::B1 + gemm::detail::BT<float> + 1;;
	constexpr int K = 1;

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

TEST_CASE("column", "[large][special]") {
	constexpr int M = gemm::detail::B2 + gemm::detail::B1 + gemm::detail::BT<float> + 1;
	constexpr int N = 1;
	constexpr int K = 1;

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
