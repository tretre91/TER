#include <random>

#include <eblas.hpp>
#include <nanobench.h>
#include <openblas/cblas.h>
#include <algorithm>
#include "util.hpp"

void naive_mm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
	std::transform(C, C + M*N, C, [&](auto x) { return beta * x; });
	for (int ay = 0; ay < M; ay++) {
		for (int ax = 0; ax < K; ax++) {
			for (int bx = 0; bx < N; bx++) {
				C[bx + ay * N] += alpha * A[ax + ay * K] * B[bx + ax * M];
			}
		}
	}
}

int main(int argc, char* argv[]) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0.f, 1000.f);

	constexpr int default_dim = 512;
	int dim = 0;
	if (argc >= 2) {
		dim = std::atoi(argv[1]);
	}
	dim = dim == 0 ? default_dim : dim;

	const std::vector<float> A = util::identity(dim);

	std::vector<float> B(dim * dim);
	for (std::size_t i = 0; i < B.size(); i++) {
		B[i] = i;
	}

	std::vector<float> C(dim * dim);	

	auto oldC = C;

	const float alpha = 1.77f;
	const float beta = 0.5f;
	eblas::sgemm(eblas::transposition::none, eblas::transposition::none, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim);
	util::check_matrix(dim, dim, dim, alpha, beta, A, B, oldC, C);
	C = oldC;
	naive_mm(dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim);
	util::check_matrix(dim, dim, dim, alpha, beta, A, B, oldC, C);

	using namespace ankerl;

	nanobench::Bench bench;
	bench.title("Matrix multiplication").relative(true);

	C = oldC;
	bench.run("my impl", [&]() { eblas::sgemm(eblas::transposition::none, eblas::transposition::none, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim); });
	C = oldC;
	bench.run("open_blas", [&]() { cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim); });
	C = oldC;
	bench.run("naive", [&]() { naive_mm(dim, dim, dim, alpha, A.data(), dim, B.data(), dim, beta, C.data(), dim); });
}
