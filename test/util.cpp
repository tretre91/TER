#include "util.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <openblas/cblas.h>

void util::print_mat(int M, int N, const float* Mat) {
	for (int i = 0; i < M; i++) {
		fmt::print("{::7.6}\n", std::span(Mat + i * N, Mat + (i + 1) * N));
	}
	fmt::print("\n");
}

bool util::check_matrix(
  int M, int N, int K, float alpha, float beta, std::span<const float> A, std::span<const float> B, std::span<const float> oldC, std::span<const float> C) {
	std::vector<float> result(oldC.begin(), oldC.end());
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), M, B.data(), K, beta, result.data(), M);
	for (std::size_t i = 0; i < C.size(); i++) {
		if (std::abs(C[i] - result[i]) > 1e-12) {
			fmt::print("Element mismatch at index {}: computed value = {:.8} (should be {:.8})\n", i, C[i], result[i]);
			print_mat(M, K, A.data());
			print_mat(K, N, B.data());
			print_mat(M, N, C.data());
			print_mat(M, N, result.data());

			return false;
		}
	}
	return true;
}

std::vector<float> util::identity(int dim) {
	std::vector<float> mat(dim * dim);
	for (int i = 0; i < dim; i++) {
		mat[i + i * dim] = 1.f;
	}
	return mat;
}

