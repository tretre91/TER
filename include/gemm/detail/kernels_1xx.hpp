#ifndef GEMM_KERNELS_1XX_HPP
#define GEMM_KERNELS_1XX_HPP

#include <eve/eve.hpp>

namespace gemm::detail
{
	///////////////////////////////////////
	//            1x1 kernels            //
	///////////////////////////////////////

	template<typename T>
	void kernel_111(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		*c += *a * *b;
	}

	template<typename T>
	void kernel_112(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		c[0] += a[0] * b[0];
		c[1] += a[0] * b[1];
	}

	template<typename T>
	void kernel_114(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		eve::store(eve::fma(*a, wide_t{b}, wide_t{c}), c);
	}

	template<typename T>
	void kernel_118(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		eve::store(eve::fma(*a, wide_t{b}, wide_t{c}), c);
	}

	///////////////////////////////////////
	//            1x2 kernels            //
	///////////////////////////////////////

	template<typename T>
	void kernel_121(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		c[0] += a[0] * b[0] + a[1] * b[stride_b];
	}

	template<typename T>
	void kernel_122(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		c[0] += a[0] * b[0] + a[1] * b[stride_b];
		c[1] += a[0] * b[1] + a[1] * b[stride_b + 1];
	}

	template<typename T>
	void kernel_124(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		wide_t wc{c};
		wc += a[0] * wide_t{b} + a[1] * wide_t{b + stride_b};
		eve::store(wc, c);
	}

	template<typename T>
	void kernel_128(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		wide_t wc{c};
		wc += a[0] * wide_t{b} + a[1] * wide_t{b + stride_b};
		eve::store(wc, c);
	}

	///////////////////////////////////////
	//            1x4 kernels            //
	///////////////////////////////////////

	template<typename T>
	void kernel_141(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		*c += eve::reduce(wide_t{a} * wide_t{b[0], b[stride_b], b[2 * stride_b], b[3 * stride_b]});
	}

	template<typename T>
	void kernel_142(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		const wide_t wa{a};
		c[0] += eve::reduce(wa * wide_t{b[0], b[stride_b], b[2 * stride_b], b[3 * stride_b]});
		c[1] += eve::reduce(wa * wide_t{b[1], b[stride_b + 1], b[2 * stride_b + 1], b[3 * stride_b + 1]}); // TODO
	}

	template<typename T>
	void kernel_144(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		wide_t wc{c};
		wc += a[0] * wide_t{b} + a[1] * wide_t{b + stride_b};
		wc += a[2] * wide_t{b + 2 * stride_b} + a[3] * wide_t{b + 3 * stride_b};
		eve::store(wc, c);
	}

	template<typename T>
	void kernel_148(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		wide_t wc{c};
		wc += a[0] * wide_t{b} + a[1] * wide_t{b + stride_b};
		wc += a[2] * wide_t{b + 2 * stride_b} + a[3] * wide_t{b + 3 * stride_b};
		eve::store(wc, c);
	}

	///////////////////////////////////////
	//            1x8 kernels            //
	///////////////////////////////////////

	template<typename T>
	void kernel_181(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		*c += eve::reduce(wide_t{a} * wide_t{[=](auto i, auto) { return b[i * stride_b]; }}); // TODO ?
	}

	template<typename T>
	void kernel_182(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		c[0] += eve::reduce(wide_t{a} * wide_t{[=](auto i, auto) { return b[i * stride_b]; }});
		c[1] += eve::reduce(wide_t{a} * wide_t{[=](auto i, auto) { return b[i * stride_b + 1]; }}); // TODO
	}

	template<typename T>
	void kernel_184(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<4>>;
		wide_t wc{c};
		wc += a[0] * wide_t{b} + a[1] * wide_t{b + stride_b};
		wc += a[2] * wide_t{b + 2 * stride_b} + a[3] * wide_t{b + 3 * stride_b};
		wc += a[4] * wide_t{b + 4 * stride_b} + a[5] * wide_t{b + 5 * stride_b};
		wc += a[6] * wide_t{b + 6 * stride_b} + a[7] * wide_t{b + 7 * stride_b};
		eve::store(wc, c);
	}

	template<typename T>
	void kernel_188(const T* a, const int stride_a, const T* b, const int stride_b, T* c, const int stride_c) {
		using wide_t = eve::wide<T, eve::fixed<8>>;
		wide_t wc{c};
		wc += a[0] * wide_t{b} + a[1] * wide_t{b + stride_b};
		wc += a[2] * wide_t{b + 2 * stride_b} + a[3] * wide_t{b + 3 * stride_b};
		wc += a[4] * wide_t{b + 4 * stride_b} + a[5] * wide_t{b + 5 * stride_b};
		wc += a[6] * wide_t{b + 6 * stride_b} + a[7] * wide_t{b + 7 * stride_b};
		eve::store(wc, c);
	}
} // namespace gemm::detail

#endif
