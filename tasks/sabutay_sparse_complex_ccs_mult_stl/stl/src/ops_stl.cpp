#include "sabutay_sparse_complex_ccs_mult_stl/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <thread>
#include <vector>

#include "sabutay_sparse_complex_ccs_mult_stl/common/include/common.hpp"

namespace sabutay_sparse_complex_ccs_mult_stl {

SabutaySparseComplexCcsMultSTL::SabutaySparseComplexCcsMultSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCS();
}

void SabutaySparseComplexCcsMultSTL::SpMM(const CCS &a, const CCS &b, CCS &c) {
  struct ColumnData {
    std::vector<int> row_ind;
    std::vector<std::complex<double>> values;
  };

  c.m = a.m;
  c.n = b.n;
  c.col_ptr.assign(b.n + 1, 0);
  c.row_ind.clear();
  c.values.clear();

  if (a.m == 0 || b.n == 0) {
    return;
  }

  constexpr double kEps = 1e-14;
  std::vector<ColumnData> columns(static_cast<std::size_t>(b.n));

  auto compute_columns = [&](int col_begin, int col_end) {
    const std::complex<double> zero(0.0, 0.0);
    std::vector<int> rows;
    std::vector<int> marker(a.m, -1);
    std::vector<std::complex<double>> acc(a.m, zero);

    for (int j = col_begin; j < col_end; ++j) {
      rows.clear();

      for (int k = b.col_ptr[j]; k < b.col_ptr[j + 1]; ++k) {
        const std::complex<double> b_value = b.values[k];
        const int b_row = b.row_ind[k];

        for (int a_index = a.col_ptr[b_row]; a_index < a.col_ptr[b_row + 1]; ++a_index) {
          const int row = a.row_ind[a_index];
          acc[row] += b_value * a.values[a_index];
          if (marker[row] != j) {
            rows.push_back(row);
            marker[row] = j;
          }
        }
      }

      ColumnData &column = columns[static_cast<std::size_t>(j)];
      for (const int row : rows) {
        if (std::abs(acc[row]) > kEps) {
          column.row_ind.push_back(row);
          column.values.push_back(acc[row]);
        }
        acc[row] = zero;
      }
    }
  };

  const unsigned int hw_threads = std::thread::hardware_concurrency();
  const int threads_count = std::max(1, std::min(b.n, static_cast<int>(hw_threads == 0U ? 1U : hw_threads)));
  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(threads_count));

  const int base_chunk = b.n / threads_count;
  const int remainder = b.n % threads_count;
  int begin = 0;
  for (int thread_index = 0; thread_index < threads_count; ++thread_index) {
    const int chunk = base_chunk + (thread_index < remainder ? 1 : 0);
    const int end = begin + chunk;
    threads.emplace_back(compute_columns, begin, end);
    begin = end;
  }
  for (std::thread &thread : threads) {
    thread.join();
  }

  for (int j = 0; j < b.n; ++j) {
    const ColumnData &column = columns[static_cast<std::size_t>(j)];
    c.col_ptr[j + 1] = c.col_ptr[j] + static_cast<int>(column.row_ind.size());
    c.row_ind.insert(c.row_ind.end(), column.row_ind.begin(), column.row_ind.end());
    c.values.insert(c.values.end(), column.values.begin(), column.values.end());
  }
}

bool SabutaySparseComplexCcsMultSTL::ValidationImpl() {
  const CCS &a = std::get<0>(GetInput());
  const CCS &b = std::get<1>(GetInput());
  return a.n == b.m;
}

bool SabutaySparseComplexCcsMultSTL::PreProcessingImpl() {
  return true;
}

bool SabutaySparseComplexCcsMultSTL::RunImpl() {
  const CCS &a = std::get<0>(GetInput());
  const CCS &b = std::get<1>(GetInput());
  CCS &c = GetOutput();
  SpMM(a, b, c);
  return true;
}

bool SabutaySparseComplexCcsMultSTL::PostProcessingImpl() {
  return true;
}

}  // namespace sabutay_sparse_complex_ccs_mult_stl
