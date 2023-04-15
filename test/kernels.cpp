#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <nanobench.h>

#include "util.hpp"
#include <gemm/detail/kernels.hpp>

using Catch::Matchers::WithinRel;

TEST_CASE("Microkernels", "[kernels]") {
	Catch::StringMaker<float>::precision = 15;

	constexpr float alpha = 1.0f;
	constexpr float beta = 1.0f;

	const auto M = GENERATE(1, 2 /*, 4, 8*/);
	const auto K = GENERATE(1, 2, 4, 8);

	const auto A = util::random_vector<float>(M * K);
	
	const auto N = GENERATE(1, 2, 4, 8);
	
	CAPTURE(M, N, K);

	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	auto C2 = C;

	// TODO: move benchmarks here?

	gemm::detail::get_kernel<float>(M, N, K)(M, N, K, A.data(), K, B.data(), N, C.data(), N);
	util::openblas_sgemm(M, N, K, alpha, A, K, B, N, beta, C2, N);

	for (std::size_t i = 0; i < C.size(); i++) {
		CAPTURE(i);
		REQUIRE_THAT(C[i], WithinRel(C2[i], util::precision<float>));
	}
}

TEST_CASE("Kernels benchmark", "[.benchmark][kernels]") {
	using namespace ankerl;
	using namespace std::chrono_literals;

	Catch::StringMaker<float>::precision = 15;

	constexpr float alpha = 1.0f;
	constexpr float beta = 1.0f;

	auto bench = nanobench::Bench().warmup(10).minEpochTime(20ms).relative(true).output(util::benchmark_output);

	const auto M = GENERATE(1, 2 /*, 4, 8*/);
	const auto K = GENERATE(1, 2, 4, 8);

	const auto A = util::random_vector<float>(M * K);

	const auto N = GENERATE(1, 2, 4, 8);

	CAPTURE(M, N, K);

	const auto B = util::random_vector<float>(K * N);
	auto C = util::random_vector<float>(M * N);
	const auto oldC = C;

	const std::string title = fmt::format("{}x{} * {}x{}", M, K, K, N);
	fmt::print(*util::benchmark_output, "\n#### {}\n", title);

	bench.title(title);
	auto* kernel = gemm::detail::get_kernel<float>(M, N, K);
	bench.run("gemm", [=, &C] { kernel(M, N, K, A.data(), K, B.data(), N, C.data(), N); });
	C = oldC;
	bench.run("naive", [=, &C] { util::naive_gemm(M, N, K, alpha, A.data(), K, B.data(), N, beta, C.data(), N); });
	C = oldC;
	bench.run("blas", [=, &C] { util::openblas_sgemm(M, N, K, alpha, A, K, B, N, beta, C, N); });
}
