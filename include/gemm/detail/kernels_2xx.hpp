#ifndef GEMM_KERNELS_2XX_HPP
#define GEMM_KERNELS_2XX_HPP

#include <eve/eve.hpp>

namespace gemm::detail
{
	///////////////////////////////////////
	//         2x1 * 1xX kernels         //
	///////////////////////////////////////

	template<typename T>
	void kernel_211(const T* a, const int stride_a, const T* b, const int /* stride_b */, T* c, const int stride_c) {
		c[0] += a[0] * *b;
		c[stride_c] += a[stride_a] * *b;
	}

	template<typename T>
	void kernel_212(const T* a, const int stride_a, const T* b, const int /* stride_b */, T* c, const int stride_c) {
		c[0] += a[0] * b[0];
		c[1] += a[0] * b[1];
		c[stride_c] += a[stride_a] * b[0];
		c[stride_c + 1] += a[stride_a] * b[1];
	}

	template<typename T>
	void kernel_214(const T* a, const int stride_a, const T* b, const int /* stride_b */, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		eve::store(eve::fma(a[0], wide_t{b}, wide_t{c}), c);
		eve::store(eve::fma(a[stride_a], wide_t{b}, wide_t{c + stride_c}), c + stride_c);
	}

	template<typename T>
	void kernel_218(const T* a, const int stride_a, const T* b, const int /* stride_b */, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		eve::store(eve::fma(a[0], wide_t{b}, wide_t{c}), c);
		eve::store(eve::fma(a[stride_a], wide_t{b}, wide_t{c + stride_c}), c + stride_c);
	}

	///////////////////////////////////////
	//         2x2 * 2xX kernels         //
	///////////////////////////////////////

	template<typename T>
	void kernel_221(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		c[0] += a[0] * b[0] + a[1] * b[stride_b];
		c[stride_c] += a[stride_a] * b[0] + a[stride_a + 1] * b[stride_b];
	}

	template<typename T>
	void kernel_222(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		const wide_t wa{a[0], a[1], a[stride_a], a[stride_a + 1]};
		const wide_t wb{b[0], b[1], b[stride_b], b[stride_b + 1]};

		const auto a0 = eve::shuffle(wa, eve::pattern<0, 0, 2, 2>);
		const auto b0 = eve::shuffle(wb, eve::pattern<0, 1, 0, 1>);
		const auto a1 = eve::shuffle(wa, eve::pattern<1, 1, 3, 3>);
		const auto b1 = eve::shuffle(wb, eve::pattern<2, 3, 2, 3>);
		const auto res = eve::fma(a0, b0, a1 * b1);

		c[0] += res.get(0);
		c[1] += res.get(1);
		c[stride_c] += res.get(2);
		c[stride_c + 1] += res.get(3);
	}

	template<typename T>
	void kernel_224(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		const wide_t wb0{b};
		const wide_t wb1{b + stride_b};

		const auto res0 = eve::fma(a[0], wb0, a[1] * wb1);
		const auto res1 = eve::fma(a[stride_a], wb0, a[stride_a + 1] * wb1);

		c[0] += res0.get(0);
		c[1] += res0.get(1);
		c[2] += res0.get(2);
		c[3] += res0.get(3);
		c[stride_c] += res1.get(0);
		c[stride_c + 1] += res1.get(1);
		c[stride_c + 2] += res1.get(2);
		c[stride_c + 3] += res1.get(3);
	}

	template<typename T>
	void kernel_228(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		const wide_t wb0{b};
		const wide_t wb1{b + stride_b};
		const wide_t wc0{c};
		const wide_t wc1{c + stride_c};

		const auto res1 = eve::fma(a[0], wb0, a[1] * wb1);
		const auto res2 = eve::fma(a[stride_a], wb0, a[stride_a + 1] * wb1);

		eve::store(res1 + wc0, c);
		eve::store(res2 + wc1, c + stride_c);
	}

	///////////////////////////////////////
	//         2x4 * 4xX kernels         //
	///////////////////////////////////////

	template<typename T>
	void kernel_241(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		const wide_t wa0{a};
		const wide_t wa1{a + stride_a};
		const wide_t wb{b[0], b[stride_b], b[2 * stride_b], b[3 * stride_b]};

		c[0] += eve::reduce(wa0 * wb);
		c[stride_c] += eve::reduce(wa1 * wb);
	}

	template<typename T>
	void kernel_242(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		const wide_t wa0{a};
		const wide_t wa1{a + stride_a};
		const wide_t wb0{b[0], b[stride_b], b[2 * stride_b], b[3 * stride_b]};
		const wide_t wb1{b[1], b[stride_b + 1], b[2 * stride_b + 1], b[3 * stride_b + 1]};

		c[0] += eve::reduce(wa0 * wb0);
		c[1] += eve::reduce(wa0 * wb1);
		c[stride_c] += eve::reduce(wa1 * wb0);
		c[stride_c + 1] += eve::reduce(wa1 * wb1);
	}

	template<typename T>
	void kernel_244(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		const wide_t wb0{b};
		const wide_t wb1{b + stride_b};
		const wide_t wb2{b + 2 * stride_b};
		const wide_t wb3{b + 3 * stride_b};

		const auto c11 = eve::fma(a[0], wb0, a[1] * wb1);
		const auto c12 = eve::fma(a[2], wb2, a[3] * wb3);
		eve::store(wide_t{c} + c11 + c12, c);

		const auto c21 = eve::fma(a[stride_a], wb0, a[stride_a + 1] * wb1);
		const auto c22 = eve::fma(a[stride_a + 2], wb2, a[stride_a + 3] * wb3);
		eve::store(wide_t{c + stride_c} + c21 + c22, c + stride_c);
	}

	template<typename T>
	void kernel_248(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		const wide_t wb0{b};
		const wide_t wb1{b + stride_b};
		const wide_t wb2{b + 2 * stride_b};
		const wide_t wb3{b + 3 * stride_b};

		const auto c11 = eve::fma(a[0], wb0, a[1] * wb1);
		const auto c12 = eve::fma(a[2], wb2, a[3] * wb3);
		eve::store(wide_t{c} + c11 + c12, c);

		const auto c21 = eve::fma(a[stride_a], wb0, a[stride_a + 1] * wb1);
		const auto c22 = eve::fma(a[stride_a + 2], wb2, a[stride_a + 3] * wb3);
		eve::store(wide_t{c + stride_c} + c21 + c22, c + stride_c);
	}

	///////////////////////////////////////
	//         2x8 * 8xX kernels         //
	///////////////////////////////////////

	template<typename T>
	void kernel_281(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		const wide_t wa1{a};
		const wide_t wa2{a + stride_a};
		const wide_t wb{[=](auto i, auto) { return b[i * stride_b]; }};

		c[0] += eve::reduce(wa1 * wb);
		c[stride_c] += eve::reduce(wa2 * wb);
	}

	template<typename T>
	void kernel_282(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		const wide_t wa0{a};
		const wide_t wa1{a + stride_a};
		// In this case, with GCC, explicitely constructing piecewise faster than using a generator for some reason
		const wide_t wb0{b[0], b[stride_b], b[2 * stride_b], b[3 * stride_b], b[4 * stride_b], b[5 * stride_b], b[6 * stride_b], b[7 * stride_b]};
		const wide_t wb1{
		  b[1], b[stride_b + 1], b[2 * stride_b + 1], b[3 * stride_b + 1], b[4 * stride_b + 1], b[5 * stride_b + 1], b[6 * stride_b + 1], b[7 * stride_b + 1]};

		c[0] += eve::reduce(wa0 * wb0);
		c[1] += eve::reduce(wa0 * wb1);
		c[stride_c] += eve::reduce(wa1 * wb0);
		c[stride_c + 1] += eve::reduce(wa1 * wb1);
	}

	template<typename T>
	void kernel_284(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;

		const wide_t wb0{b};
		const wide_t wb1{b + stride_b};
		const wide_t wb2{b + 2 * stride_b};
		const wide_t wb3{b + 3 * stride_b};
		const wide_t wb4{b + 4 * stride_b};
		const wide_t wb5{b + 5 * stride_b};
		const wide_t wb6{b + 6 * stride_b};
		const wide_t wb7{b + 7 * stride_b};

		const auto res11 = eve::fma(a[0], wb0, a[1] * wb1);
		const auto res12 = eve::fma(a[2], wb2, a[3] * wb3);
		const auto res13 = eve::fma(a[4], wb4, a[5] * wb5);
		const auto res14 = eve::fma(a[6], wb6, a[7] * wb7);

		const auto res21 = eve::fma(a[stride_a], wb0, a[stride_a + 1] * wb1);
		const auto res22 = eve::fma(a[stride_a + 2], wb2, a[stride_a + 3] * wb3);
		const auto res23 = eve::fma(a[stride_a + 4], wb4, a[stride_a + 5] * wb5);
		const auto res24 = eve::fma(a[stride_a + 6], wb6, a[stride_a + 7] * wb7);

		eve::store(wide_t{c} + res11 + res12 + res13 + res14, c);
		eve::store(wide_t{c + stride_c} + res21 + res22 + res23 + res24, c + stride_c);
	}

	template<typename T>
	void kernel_288(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;

		const wide_t wb0{b};
		const wide_t wb1{b + stride_b};
		const wide_t wb2{b + 2 * stride_b};
		const wide_t wb3{b + 3 * stride_b};
		const wide_t wb4{b + 4 * stride_b};
		const wide_t wb5{b + 5 * stride_b};
		const wide_t wb6{b + 6 * stride_b};
		const wide_t wb7{b + 7 * stride_b};

		const auto res11 = eve::fma(a[0], wb0, a[1] * wb1);
		const auto res12 = eve::fma(a[2], wb2, a[3] * wb3);
		const auto res13 = eve::fma(a[4], wb4, a[5] * wb5);
		const auto res14 = eve::fma(a[6], wb6, a[7] * wb7);

		const auto res21 = eve::fma(a[stride_a], wb0, a[stride_a + 1] * wb1);
		const auto res22 = eve::fma(a[stride_a + 2], wb2, a[stride_a + 3] * wb3);
		const auto res23 = eve::fma(a[stride_a + 4], wb4, a[stride_a + 5] * wb5);
		const auto res24 = eve::fma(a[stride_a + 6], wb6, a[stride_a + 7] * wb7);

		eve::store(wide_t{c} + res11 + res12 + res13 + res14, c);
		eve::store(wide_t{c + stride_c} + res21 + res22 + res23 + res24, c + stride_c);
	}
} // namespace gemm::detail

#endif
