#ifndef GEMM_GEMM_HPP
#define GEMM_GEMM_HPP

#include "gemm/detail/kernels.hpp"
#include <eve/eve.hpp>
#include <eve/module/algo.hpp>
#include <eve/module/core.hpp>

#include <array>
#include <span>
#include <vector>

namespace gemm
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

		template<typename T>
		inline constexpr void multiply_tile(const tile<T>& a, const tile<T>& b, T* dest, const int ld_dest) {
			auto c = load_tile(dest, ld_dest);
			compute_tile(a, b, c);
			store_tile(c, dest, ld_dest);
		}

		static constexpr int B1 = 64;
		static constexpr int B2 = 256;

		template<typename T>
		void compute_b1_block(const int i, const int k, const int j, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			constexpr auto tile_size = eve::wide<T>::size();

			for (int tile_i = i; tile_i < i + B1; tile_i += tile_size) {
				for (int tile_k = k; tile_k < k + B1; tile_k += tile_size) {
					const auto a_tile = detail::load_tile(&A[tile_k + lda * tile_i], lda);
					for (int tile_j = j; tile_j < j + B1; tile_j += tile_size) {
						multiply_tile(a_tile, detail::load_tile(&B[tile_j + ldb * tile_k], ldb), &C[tile_j + ldc * tile_i], ldc);
					}
				}
			}
		}

		template<typename T>
		void compute_b2_block(const int i, const int k, const int j, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			for (int block_i = i; block_i < i + B2; block_i += B1) {
				for (int block_k = k; block_k < k + B2; block_k += B1) {
					for (int block_j = j; block_j < j + B2; block_j += B1) {
						compute_b1_block(block_i, block_k, block_j, A, lda, B, ldb, C, ldc);
					}
				}
			}
		}
	} // namespace detail

	inline void sgemm(transposition transA, transposition transB, const int M, const int N, const int K, const float alpha, const float* A, const int lda,
	  const float* B, const int ldb, const float beta, float* C, const int ldc) {
		using detail::B1;
		using detail::B2;

		std::vector<float> ab(M * N);

		for (int i = 0; i < M; i += B2) {
			for (int k = 0; k < K; k += B2) {
				for (int j = 0; j < N; j += B2) {
					detail::compute_b2_block(i, k, j, A, lda, B, ldb, ab.data(), ldc); // B2 blocks // TODO: use lda, ldb, ldc
				}
			}
		}

		auto c = std::span(C, M * N);
		eve::algo::transform_to(eve::views::zip(ab, c), c, [alpha, beta](auto x) { return alpha * eve::get<0>(x) + beta * eve::get<1>(x); });
	}
} // namespace gemm

#endif
