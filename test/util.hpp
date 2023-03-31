#ifndef TEST_UTIL_HPP
#define TEST_UTIL_HPP

#include <span>
#include <vector>

namespace util
{
	void print_mat(int M, int N, const float* mat);

	bool check_matrix(
	  int M, int N, int K, float alpha, float beta, std::span<const float> A, std::span<const float> B, std::span<const float> oldC, std::span<const float> C);

	std::vector<float> identity(int dim);
}; // namespace util

#endif
