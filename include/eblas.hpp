#ifndef EBLAS_HPP
#define EBLAS_HPP

#include <eve/eve.hpp>
#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

#include <numeric>
#include <ranges>
#include <span>
#include <vector>


namespace eblas
{
	enum class transposition {
		none,
		transpose,
		conjugate_transpose
	};

	namespace detail
	{
		template<typename T>
		using tile = std::array<eve::wide<T>, eve::wide<T>::size()>;

		template<typename T>
		tile<T> load_tile(const T* addr, const std::size_t stride) {
			using wide_t = eve::wide<T>;
			std::array<wide_t, wide_t::size()> tile;
			for (auto& line : tile) {
				line = wide_t{addr};
				addr += stride;
			}
			return tile;
		}

		template<typename T>
		void store_tile(const tile<T>& tile, T* addr, const std::size_t stride) {
			for (auto line : tile) {
				eve::store(line, addr);
				addr += stride;
			}
		}

		template<typename T>
		void compute_tile(const tile<T>& a_tile, const tile<T>& b_tile, tile<T>& c_tile) {
			using wide_t = eve::wide<T>;
			for (std::size_t i = 0; i < c_tile.size(); i++) {
				for (std::size_t x = 0; x < wide_t::size(); x++) {
					c_tile[i] += a_tile[i].get(x) * b_tile[x];
				}
			}
		}
	} // namespace detail


	inline void sgemm(transposition transA, transposition transB, const int M, const int N, const int K, const float alpha, const float* A, const int lda,
	  const float* B, const int ldb, const float beta, float* C, const int ldc) {
		using wide_t = eve::wide<float>;
		constexpr auto tile_size = wide_t::size();

		std::vector<float> ab(M * N);

		for (int block_ay = 0; block_ay < M; block_ay += tile_size) {
			for (int block_k = 0; block_k < K; block_k += tile_size) {
				for (int block_bx = 0; block_bx < N; block_bx += tile_size) {
					const auto a_tile = detail::load_tile(&A[block_k + K * block_ay], K);
					const auto b_tile = detail::load_tile(&B[block_bx + N * block_k], N);
					auto c_tile = detail::load_tile(&ab[block_bx + N * block_ay], N);

					detail::compute_tile(a_tile, b_tile, c_tile);

					detail::store_tile(c_tile, &ab[block_bx + N * block_ay], N);
				}
			}
		}

		auto c = std::span(C, M * N);
		eve::algo::transform_to(eve::views::zip(ab, c), c, [alpha, beta](auto x) { return alpha * eve::get<0>(x) + beta * eve::get<1>(x); });
	}
} // namespace eblas

#endif
