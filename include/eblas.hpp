#ifndef EBLAS_HPP
#define EBLAS_HPP

#include <eve/eve.hpp>
#include <eve/module/core.hpp>
#include <eve/module/algo.hpp>
#include <span>
#include <utility>
#include <ranges>
#include <numeric>
#include <vector>

template<std::size_t N>
constexpr auto unroll_impl = [](auto expr) {
	[expr]<auto ...Is>(std::index_sequence<Is...>) {
		(expr(Is), ...);
	} (std::make_index_sequence<N>{});
};

template<auto start, auto end, auto N>
constexpr auto unroll(auto expr) {
	for (std::size_t i = start; i < end; i += N) {
		[expr, i]<auto ...Is>(std::index_sequence<Is...>) {
			(expr(Is + i), ...);
		} (std::make_index_sequence<N>{});
	}
}

namespace eblas
{
	enum class transposition {
		none,
		transpose,
		conjugate_transpose
	};

	inline void sgemm(transposition transA, transposition transB, const int M, const int N, const int K, const float alpha, const float* A, const int lda,
	  const float* B, const int ldb, const float beta, float* C, const int ldc) {
		using wide_t = eve::wide<float>;
		constexpr auto tile_size = wide_t::size();

		std::vector<float> temp(M * N);
		std::array<wide_t, tile_size> lines;

		for (int block_ay = 0; block_ay < M; block_ay += tile_size) {
		for (int block_k = 0; block_k < K; block_k += tile_size) {
		for (int block_bx = 0; block_bx < N; block_bx += tile_size) {
			for (std::size_t i = 0; i < lines.size(); i++) {
				lines[i] = wide_t{&B[block_bx + N * (block_k + i)]};
			}

			for (std::size_t i = 0; i < tile_size; i++) {
				wide_t c{&temp[block_bx + N * (block_ay + i)]};
				for (std::size_t x = 0; x < tile_size; x+=4) {
					c += A[block_k + x + K * (block_ay + i)] * lines[x]
					   + A[block_k + x+1 + K * (block_ay + i)] * lines[x+1]
					   + A[block_k + x+2 + K * (block_ay + i)] * lines[x+2]
					   + A[block_k + x+3 + K * (block_ay + i)] * lines[x+3];
				}
				// c = std::transform_reduce(lines.begin(), lines.end(), &A[block_k + K * (block_ay + i)], c, std::plus<>(), std::multiplies<>());
				// c = std::accumulate(lines.begin(), lines.end(), c, [&,x=0](auto acc, auto w) mutable { return eve::fma(A[block_k + x++ + K * (block_ay + i)], w, acc); });
				// unroll<0, vec_tile_size, 8>([&](auto idx) {c = eve::fma(A[block_k + idx + K * (block_ay + i)], lines[idx], c); });
				eve::store(c, &temp[block_bx + N * (block_ay + i)]);
			}
		}
		}
		}

		auto cs = std::span(C, M * N);
		eve::algo::transform_to(eve::views::zip(temp, cs), cs, [alpha, beta](auto x) { return alpha * eve::get<0>(x) + beta * eve::get<1>(x); });
		// eve::algo::transform_to(eve::views::zip(temp, cs), cs, [alpha, beta](auto x) { return eve::sum_of_prod(alpha, eve::get<0>(x), beta, eve::get<1>(x)); });
	}
} // namespace eblas

#endif
