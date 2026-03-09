#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"

namespace redkina_a_integral_simpson_seq {

namespace {

inline int GetWeight(int idx, int n) {
  if (idx == 0 || idx == n) return 1;
  return (idx % 2 == 1) ? 4 : 2;
}

}  // namespace

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) return false;
  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) return false;
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) return false;
  }
  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();
  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;
  result_ = 0.0;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  const size_t dim = a_.size();

  std::vector<double> a = a_;
  std::vector<double> b = b_;
  std::vector<int> n = n_;
  std::vector<double> h(dim);
  for (size_t i = 0; i < dim; ++i) h[i] = (b[i] - a[i]) / static_cast<double>(n[i]);

  std::vector<size_t> strides(dim);
  strides[dim - 1] = 1;
  for (int i = static_cast<int>(dim) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * static_cast<size_t>(n[i + 1] + 1);

  const size_t total_nodes = [&] {
    size_t prod = 1;
    for (int ni : n) prod *= static_cast<size_t>(ni + 1);
    return prod;
  }();

  const double *a_data = a.data();
  const double *h_data = h.data();
  const int *n_data = n.data();
  const size_t *strides_data = strides.data();

  double sum = 0.0;

#pragma omp parallel
  {
    std::vector<int> indices(dim);
    std::vector<double> point(dim);

#pragma omp for reduction(+ : sum)
    for (size_t linear_idx = 0; linear_idx < total_nodes; ++linear_idx) {
      size_t remainder = linear_idx;
      for (size_t i = 0; i < dim; ++i) {
        indices[i] = static_cast<int>(remainder / strides_data[i]);
        remainder %= strides_data[i];
      }

      double w_prod = 1.0;
      for (size_t i = 0; i < dim; ++i) {
        w_prod *= static_cast<double>(GetWeight(indices[i], n_data[i]));
        point[i] = a_data[i] + static_cast<double>(indices[i]) * h_data[i];
      }

      sum += w_prod * func_(point);
    }
  }

  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) h_prod *= h[i];

  double denominator = 1.0;
  for (size_t i = 0; i < dim; ++i) denominator *= 3.0;

  result_ = (h_prod / denominator) * sum;
  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq
