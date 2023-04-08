#ifndef EBLAS_HPP
#define EBLAS_HPP

#include <eve/eve.hpp>
#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

#include <ranges>
#include <array>
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
				for (std::size_t x = 0; x < wide_t::size(); x += 2) {
					c_tile[i] += a_tile[i].get(x) * b_tile[x] + a_tile[i].get(x + 1) * b_tile[x + 1];
				}
			}
		}
	} // namespace detail

	inline void sgemm(transposition transA, transposition transB, const int M, const int N, const int K, const float alpha, const float* A, const int lda,
	  const float* B, const int ldb, const float beta, float* C, const int ldc) {
		constexpr auto tile_size = eve::wide<float>::size();
		constexpr int B2 = 256;
		constexpr int B1 = 64;

		std::vector<float> ab(M * N);

		for (int i2 = 0; i2 < M; i2 += B2) {
		for (int k2 = 0; k2 < K; k2 += B2) {
		for (int j2 = 0; j2 < N; j2 += B2) {
			for (int i1 = i2; i1 < i2 + B2; i1 += B1) {
			for (int k1 = k2; k1 < k2 + B2; k1 += B1) {
			for (int j1 = j2; j1 < j2 + B2; j1 += B1) {
				for (int i = i1; i < i1 + B1; i += tile_size) {
					for (int k = k1; k < k1 + B1; k += tile_size) {
						const auto a_tile = detail::load_tile(&A[k + K * i], K);
						for (int j = j1; j < j1 + B1; j += tile_size) {
							const auto b_tile = detail::load_tile(&B[j + N * k], N);
							auto c_tile = detail::load_tile(&ab[j + N * i], N);

							detail::compute_tile(a_tile, b_tile, c_tile);

							detail::store_tile(c_tile, &ab[j + N * i], N);
						}
					}
				}
			}
			}
			}
		}
		}
		}

		auto c = std::span(C, M * N);
		eve::algo::transform_to(eve::views::zip(ab, c), c, [alpha, beta](auto x) { return alpha * eve::get<0>(x) + beta * eve::get<1>(x); });
	}
} // namespace eblas

#endif
