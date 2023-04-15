#ifndef GEMM_KERNELS_HPP
#define GEMM_KERNELS_HPP

#include <algorithm>
#include <array>

#include "kernels_1xx.hpp"
#include "kernels_2xx.hpp"

namespace gemm::detail
{
	template<typename T>
	using kernel = void (*)(int, int, int, const T*, int, const T*, int, T*, int);

	template<typename T>
	consteval std::array<kernel<T>, 889> generate_kernel_table() {
		std::array<kernel<T>, 889> kernels;
		std::ranges::fill(kernels, nullptr);

		kernels[111] = &kernel_111;
		kernels[112] = &kernel_112;
		kernels[114] = &kernel_114;
		kernels[118] = &kernel_118;
		kernels[121] = &kernel_121;
		kernels[122] = &kernel_122;
		kernels[124] = &kernel_124;
		kernels[128] = &kernel_128;
		kernels[141] = &kernel_141;
		kernels[142] = &kernel_142;
		kernels[144] = &kernel_144;
		kernels[148] = &kernel_148;
		kernels[181] = &kernel_181;
		kernels[182] = &kernel_182;
		kernels[184] = &kernel_184;
		kernels[188] = &kernel_188;
		kernels[211] = &kernel_211;
		kernels[212] = &kernel_212;
		kernels[214] = &kernel_214;
		kernels[218] = &kernel_218;
		kernels[221] = &kernel_221;
		kernels[222] = &kernel_222;
		kernels[224] = &kernel_224;
		kernels[228] = &kernel_228;
		kernels[241] = &kernel_241;
		kernels[242] = &kernel_242;
		kernels[244] = &kernel_244;
		kernels[248] = &kernel_248;
		kernels[281] = &kernel_281;
		kernels[282] = &kernel_282;
		kernels[284] = &kernel_284;
		kernels[288] = &kernel_288;
		/*
		kernels[411] = &kernel_411;
		kernels[412] = &kernel_412;
		kernels[414] = &kernel_414;
		kernels[418] = &kernel_418;
		kernels[421] = &kernel_421;
		kernels[422] = &kernel_422;
		kernels[424] = &kernel_424;
		kernels[428] = &kernel_428;
		kernels[441] = &kernel_441;
		kernels[442] = &kernel_442;
		kernels[444] = &kernel_444;
		kernels[448] = &kernel_448;
		kernels[481] = &kernel_481;
		kernels[482] = &kernel_482;
		kernels[484] = &kernel_484;
		kernels[488] = &kernel_488;
		kernels[811] = &kernel_811;
		kernels[812] = &kernel_812;
		kernels[814] = &kernel_814;
		kernels[818] = &kernel_818;
		kernels[821] = &kernel_821;
		kernels[822] = &kernel_822;
		kernels[824] = &kernel_824;
		kernels[828] = &kernel_828;
		kernels[841] = &kernel_841;
		kernels[842] = &kernel_842;
		kernels[844] = &kernel_844;
		kernels[848] = &kernel_848;
		kernels[881] = &kernel_881;
		kernels[882] = &kernel_882;
		kernels[884] = &kernel_884;
		kernels[888] = &kernel_888; */

		return kernels;
	}

	// TODO: faire 1, 2, 4, 8 pour l'instant
	// regarder l'exemple de code reçu par mail pour le 2x2
	template<typename T>
	kernel<T> get_kernel(const int M, const int N, const int K) {
		static constexpr auto table = generate_kernel_table<T>();
		return table[M * 100 + K * 10 + N];
	}
} // namespace gemm::detail

#endif
