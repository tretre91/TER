#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <openblas/cblas.h>

#include <gemm/gemm.hpp>

#include "util.hpp"

using Catch::Matchers::WithinRel;

TEST_CASE("dim < TILE_SIZE", "[square][small][alpha=1]") {
	const auto TILE_SIZE = gemm::detail::BT<float>;
	auto dim = GENERATE_COPY(range(1, TILE_SIZE));
	
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

TEST_CASE("dim = TILE_SIZE", "[square][small][alpha=1]") {
	const float dim = gemm::detail::BT<float>;

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

TEST_CASE("dim = B1", "[square][small][alpha=1]") {
	const float dim = gemm::detail::B1;

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

TEST_CASE("dim = B2", "[square][small][alpha=1]") {
	const float dim = gemm::detail::B2;

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

TEST_CASE("dim > B2", "[square][large][alpha=1]") {
	using gemm::detail::B2;
	auto dim = GENERATE(2*B2, 4*B2, 8*B2);
	
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
