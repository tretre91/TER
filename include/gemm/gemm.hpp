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
		static constexpr int BT = eve::wide<T>::size();

		template<typename T>
		void kernel_B1_B1_B1(const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			for (int i = 0; i < B1; i += BT<T>) {
				for (int k = 0; k < B1; k += BT<T>) {
					const auto a_tile = detail::load_tile(&A[k + lda * i], lda);
					for (int j = 0; j < B1; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), &C[j + ldc * i], ldc);
					}
				}
			}
		}

		template<typename T>
		void kernel_B2_B2_B2(const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			for (int i = 0; i < B2; i += B1) {
				for (int k = 0; k < B2; k += B1) {
					for (int j = 0; j < B2; j += B1) {
						kernel_B1_B1_B1(A + i * lda + k, lda, B + k * ldb + j, ldb, C + i * ldc + j, ldc);
					}
				}
			}
		}

		///////////////////////////////////////////////////////////////////

		// Handles the multiplication of the last K columns of A with the last K lines of B (K < B1)
		template<typename T>
		void kernel_B1_K_B1(const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_K = K % BT<T>;
			const int k_lim = K - remaining_K;
			auto kernel = get_kernel<T>(BT<T>, BT<T>, remaining_K);

			for (int i = 0; i < B1; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < B1; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
				}

				if (remaining_K > 0) {
					for (int j = 0; j < B1; j += BT<T>) {
						kernel(base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}
		}

		// Handles the multiplication of the last K columns of A with the last K lines of B (K < B2)
		template<typename T>
		void kernel_B2_K_B2(const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_K = K % B1;
			const int k_lim = K - remaining_K;

			for (int i = 0; i < B2; i += B1) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += B1) {
					for (int j = 0; j < B2; j += B1) {
						kernel_B1_B1_B1(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
				}

				if (remaining_K > 0) {
					for (int j = 0; j < B2; j += B1) {
						kernel_B1_K_B1(remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}
		}

		///////////////////////////////////////////////////////////////////

		// Multiplies a B1 tile with the last N columns of a line (N < B1)
		template<typename T>
		void kernel_B1_B1_N(const int N, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_N = N % BT<T>;
			const int j_lim = N - remaining_N;
			auto kernel = get_kernel<T>(BT<T>, remaining_N, BT<T>);

			for (int i = 0; i < B1; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < B1; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < j_lim; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel(base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		// Multiplies a B2 tile with the last N columns of a line (N < B2)
		template<typename T>
		void kernel_B2_B2_N(const int N, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_N = N % B1;
			const int j_lim = N - remaining_N;

			for (int i = 0; i < B2; i += B1) {
				for (int k = 0; k < B2; k += B1) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_B1_B1_B1(A + i * lda + k, lda, B + k * ldb + j, ldb, C + i * ldc + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_B1_B1_N(remaining_N, A + i * lda + k, lda, B + k * ldb + j_lim, ldb, C + i * ldc + j_lim, ldc);
					}
				}
			}
		}

		///////////////////////////////////////////////////////////////////

		template<typename T>
		void kernel_B1_K_N(const int N, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_N = N % BT<T>;
			const int j_lim = N - remaining_N;

			const int remaining_K = K % BT<T>;
			const int k_lim = K - remaining_K;

			auto kernel_BT_K_N = get_kernel<T>(BT<T>, remaining_N, remaining_K);
			auto kernel_BT_BT_N = get_kernel<T>(BT<T>, remaining_N, BT<T>);
			auto kernel_BT_K_BT = get_kernel<T>(BT<T>, BT<T>, remaining_K);

			for (int i = 0; i < B1; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < j_lim; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_BT_BT_N(base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}

				if (remaining_K > 0) {
					for (int j = 0; j < j_lim; j += BT<T>) {
						kernel_BT_K_BT(base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}

					if (remaining_N > 0) {
						kernel_BT_K_N(base_A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		// Handles the multiplication of the last K columns of A with the lower right corner of B (K < B2, N < B2)
		template<typename T>
		void kernel_B2_K_N(const int N, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_K = K % B1;
			const int k_lim = K - remaining_K;

			const int remaining_N = N % B1;
			const int j_lim = N - remaining_N;

			for (int i = 0; i < B2; i += B1) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += B1) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_B1_B1_B1(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_B1_B1_N(remaining_N, base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_B1_K_B1(remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_B1_K_N(remaining_N, remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		///////////////////////////////////////////////////////////////////

		template<typename T>
		void kernel_M_B1_B1(const int M, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % BT<T>;
			const int i_lim = M - remaining_M;
			auto kernel = get_kernel<T>(remaining_M, BT<T>, BT<T>);

			for (int i = 0; i < i_lim; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < B1; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < B1; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < B1; k += BT<T>) {
					for (int j = 0; j < B1; j += BT<T>) {
						kernel(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldb);
					}
				}
			}
		}

		template<typename T>
		void kernel_M_B2_B2(const int M, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % B1;
			const int i_lim = M - remaining_M;

			for (int i = 0; i < i_lim; i += B1) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < B2; k += B1) {
					for (int j = 0; j < B2; j += B1) {
						kernel_B1_B1_B1(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < B2; k += B1) {
					for (int j = 0; j < B2; j += B1) {
						kernel_M_B1_B1(remaining_M, base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}
		}

		template<typename T>
		void kernel_M_B1_N(const int M, const int N, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % BT<T>;
			const int i_lim = M - remaining_M;

			const int remaining_N = N % BT<T>;
			const int j_lim = N - remaining_N;

			auto kernel_BT_BT_N = get_kernel<T>(BT<T>, remaining_N, BT<T>);
			auto kernel_M_BT_BT = get_kernel<T>(remaining_M, BT<T>, BT<T>);
			auto kernel_M_BT_N = get_kernel<T>(remaining_M, remaining_N, BT<T>);

			for (int i = 0; i < i_lim; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < B1; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < j_lim; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_BT_BT_N(base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < B1; k += BT<T>) {
					for (int j = 0; j < j_lim; j += BT<T>) {
						kernel_M_BT_BT(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_M_BT_N(base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		template<typename T>
		void kernel_M_B2_N(const int M, const int N, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % B1;
			const int i_lim = M - remaining_M;

			const int remaining_N = N % B1;
			const int j_lim = N - remaining_N;

			for (int i = 0; i < i_lim; i += B1) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < B2; k += B1) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_B1_B1_B1(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_B1_B1_N(remaining_N, base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < B2; k += B1) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_M_B1_B1(remaining_M, base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_M_B1_N(remaining_M, remaining_N, base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		template<typename T>
		void kernel_M_K_B1(const int M, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % BT<T>;
			const int i_lim = M - remaining_M;

			const int remaining_K = K % BT<T>;
			const int k_lim = K - remaining_K;

			auto kernel_BT_K_BT = get_kernel<T>(BT<T>, BT<T>, remaining_K);
			auto kernel_M_BT_BT = get_kernel<T>(remaining_M, BT<T>, BT<T>);
			auto kernel_M_K_BT = get_kernel<T>(remaining_M, BT<T>, remaining_K);


			for (int i = 0; i < i_lim; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < B1; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < B1; j += BT<T>) {
						kernel_BT_K_BT(base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < k_lim; k += BT<T>) {
					for (int j = 0; j < B1; j += BT<T>) {
						kernel_M_BT_BT(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < B1; j += BT<T>) {
						kernel_M_K_BT(base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}
		}


		template<typename T>
		void kernel_M_K_B2(const int M, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % B1;
			const int i_lim = M - remaining_M;

			const int remaining_K = K % B1;
			const int k_lim = K - remaining_K;

			for (int i = 0; i < i_lim; i += B1) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += B1) {
					for (int j = 0; j < B2; j += B1) {
						kernel_B1_B1_B1(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < B2; j += B1) {
						kernel_B1_K_B1(remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < k_lim; k += B1) {
					for (int j = 0; j < B2; j += B1) {
						kernel_M_B1_B1(remaining_M, base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < B2; j += B1) {
						kernel_M_K_B1(remaining_M, remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
				}
			}
		}

		template<typename T>
		void kernel_M_K_N_subB1(const int M, const int N, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % BT<T>;
			const int i_lim = M - remaining_M;

			const int remaining_N = N % BT<T>;
			const int j_lim = N - remaining_N;

			const int remaining_K = K % BT<T>;
			const int k_lim = K - remaining_K;

			auto kernel_BT_BT_N = get_kernel<T>(BT<T>, remaining_N, BT<T>);
			auto kernel_BT_K_BT = get_kernel<T>(BT<T>, BT<T>, remaining_K);
			auto kernel_BT_K_N = get_kernel<T>(BT<T>, remaining_N, remaining_K);


			for (int i = 0; i < i_lim; i += BT<T>) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += BT<T>) {
					const auto a_tile = detail::load_tile(base_A + k, lda);
					for (int j = 0; j < j_lim; j += BT<T>) {
						multiply_tile(a_tile, detail::load_tile(&B[j + ldb * k], ldb), base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_BT_BT_N(base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < j_lim; j += BT<T>) {
						kernel_BT_K_BT(base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_BT_K_N(base_A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				auto kernel_M_BT_BT = get_kernel<T>(remaining_M, BT<T>, BT<T>);
				auto kernel_M_BT_N = get_kernel<T>(remaining_M, remaining_N, BT<T>);
				auto kernel_M_K_BT = get_kernel<T>(remaining_M, BT<T>, remaining_K);
				auto kernel_M_K_N = get_kernel<T>(remaining_M, remaining_N, remaining_K);

				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < k_lim; k += BT<T>) {
					for (int j = 0; j < j_lim; j += BT<T>) {
						kernel_M_BT_BT(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_M_BT_N(base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < j_lim; j += BT<T>) {
						kernel_M_K_BT(base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_M_K_N(base_A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		template<typename T>
		void kernel_M_K_N_subB2(const int M, const int N, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_M = M % B1;
			const int i_lim = M - remaining_M;

			const int remaining_N = N % B1;
			const int j_lim = N - remaining_N;

			const int remaining_K = K % B1;
			const int k_lim = K - remaining_K;

			for (int i = 0; i < i_lim; i += B1) {
				const T* base_A = A + i * lda;
				T* base_C = C + i * ldc;
				for (int k = 0; k < k_lim; k += B1) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_B1_B1_B1(base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_B1_B1_N(remaining_N, base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
				if (remaining_K > 0) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_B1_K_B1(remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_B1_K_N(remaining_N, remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}

			if (remaining_M > 0) {
				const T* base_A = A + i_lim * lda;
				T* base_C = C + i_lim * ldc;
				for (int k = 0; k < k_lim; k += B1) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_M_B1_B1(remaining_M, base_A + k, lda, B + k * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_M_B1_N(remaining_M, remaining_N, base_A + k, lda, B + k * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
				if (remaining_K) {
					for (int j = 0; j < j_lim; j += B1) {
						kernel_M_K_B1(remaining_M, remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j, ldb, base_C + j, ldc);
					}
					if (remaining_N > 0) {
						kernel_M_K_N_subB1(remaining_M, remaining_N, remaining_K, base_A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, base_C + j_lim, ldc);
					}
				}
			}
		}

		// Function handling the multiplication of the last M lines of A with B (M < B2)
		template<typename T>
		void gemm_M_subB2(const int M, const int N, const int K, const T* A, const int lda, const T* B, const int ldb, T* C, const int ldc) {
			const int remaining_K = K % B2;
			const int k_lim = K - remaining_K;

			const int remaining_N = N % B2;
			const int j_lim = N - remaining_N;

			for (int k = 0; k < k_lim; k += B2) {
				for (int j = 0; j < j_lim; j += B2) {
					kernel_M_B2_B2(M, A + k, lda, B + k * ldb + j, ldb, C + j, ldc);
				}
				if (remaining_N > 0) {
					kernel_M_B2_N(M, remaining_N, A + k, lda, B + k * ldb + j_lim, ldb, C + j_lim, ldc);
				}
			}

			if (remaining_K > 0) {
				for (int j = 0; j < j_lim; j += B2) {
					kernel_M_K_B2(M, remaining_K, A + k_lim, lda, B + k_lim * ldb + j, ldb, C + j, ldc);
				}

				if (remaining_N > 0) {
					kernel_M_K_N_subB2(M, remaining_N, remaining_K, A + k_lim, lda, B + k_lim * ldb + j_lim, ldb, C + j_lim, ldc);
				}
			}
		}
	} // namespace detail

	template<typename T>
	void gemm(transposition transA, transposition transB, const int M, const int N, const int K, const T alpha, const T* A, const int lda, const T* B,
	  const int ldb, const T beta, T* C, const int ldc) {
		using detail::B1;
		using detail::B2;

		std::vector<T> ab(M * N);

		const int remaining_M = M % B2;
		const int i_lim = M - remaining_M;

		const int remaining_N = N % B2;
		const int j_lim = N - remaining_N;

		const int remaining_K = K % B2;
		const int k_lim = K - remaining_K;

		for (int i = 0; i < i_lim; i += B2) {
			const T* base_A = A + i * lda;
			T* base_AB = ab.data() + i * ldc;
			for (int k = 0; k < k_lim; k += B2) {
				const T* A = base_A + k; // TODO: utiliser les adresses comme itérateurs ?
				const T* base_B = B + k * ldb;
				for (int j = 0; j < j_lim; j += B2) {
					detail::kernel_B2_B2_B2(A, lda, base_B + j, ldb, base_AB + j, ldc);
				}
				if (remaining_N != 0) {
					detail::kernel_B2_B2_N(remaining_N, A, lda, B + k * ldb + j_lim, ldb, base_AB + j_lim, ldc);
				}
			}
			if (remaining_K > 0) {
				const T* A = base_A + k_lim;
				const T* base_B = B + k_lim * ldb; // peut être sorti de la boucle
				for (int j = 0; j < j_lim; j += B2) {
					detail::kernel_B2_K_B2(remaining_K, A, lda, base_B + j, ldb, base_AB + j, ldc);
				}
				if (remaining_N > 0) {
					detail::kernel_B2_K_N(remaining_N, remaining_K, A, lda, base_B + j_lim, ldb, base_AB + j_lim, ldc);
				}
			}
		}

		if (remaining_M > 0) {
			detail::gemm_M_subB2(remaining_M, N, K, A + i_lim * lda, lda, B, ldb, ab.data() + i_lim * ldc, ldc);
		}

		// TODO: ne fonctionne que si ldc == N :))
		// Déplacer après chaque calcul de ligne B2, devrait être correct ET cache friendly :))
		// + permet de réduire la taille de ab (besoin de stocker qu'une seule ligne B2 à la fois)
		auto c = std::span(C, M * N);
		eve::algo::transform_to(eve::views::zip(ab, c), c, [alpha, beta](auto x) { return alpha * eve::get<0>(x) + beta * eve::get<1>(x); });
	}

	inline auto sgemm = gemm<float>;
	inline auto dgemm = gemm<double>;

} // namespace gemm

#endif
