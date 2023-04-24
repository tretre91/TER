#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openblas/cblas.h>

#include <gemm/gemm.hpp>

#include "util.hpp"

using Catch::Matchers::WithinRel;

constexpr auto B2 = gemm::detail::B2<float>;
constexpr auto B1 = gemm::detail::B1<float>;
constexpr auto TILE_SIZE = gemm::detail::TILE_SIZE<float>;

TEST_CASE("worst case", "[square][large][special]") {
	constexpr auto dim = B2 + B1 + TILE_SIZE + 1;

	const auto A = util::random_vector<float>(dim * dim);
	const auto B = util::random_vector<float>(dim * dim);
	auto C = util::random_vector<float>(dim * dim);
	auto C2 = C;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	gemm::sgemm(util::no_trans, util::no_trans, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, A.data(), dim, B.data(), dim,
	  beta, C2.data(), dim);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

TEST_CASE("dot product", "[large][special]") {
	constexpr int M = 1;
	constexpr int N = 1;
	constexpr int K = B2 + B1 + TILE_SIZE + 1;

	const auto A = util::random_vector<float>(M * K);
	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	auto C2 = C;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	gemm::sgemm(util::no_trans, util::no_trans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C2.data(), N);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

TEST_CASE("outer product", "[large][special]") {
	constexpr int M = B2 + B1 + TILE_SIZE + 1;;
	constexpr int N = B2 + B1 + TILE_SIZE + 1;;
	constexpr int K = 1;

	const auto A = util::random_vector<float>(M * K);
	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	auto C2 = C;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	gemm::sgemm(util::no_trans, util::no_trans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C2.data(), N);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

TEST_CASE("column", "[large][special]") {
	constexpr int M = B2 + B1 + TILE_SIZE + 1;
	constexpr int N = 1;
	constexpr int K = 1;

	const auto A = util::random_vector<float>(M * K);
	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	auto C2 = C;

	const float alpha = util::random_float<float>();
	const float beta = util::random_float<float>();

	gemm::sgemm(util::no_trans, util::no_trans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), K, B.data(), N, beta, C2.data(), N);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}
