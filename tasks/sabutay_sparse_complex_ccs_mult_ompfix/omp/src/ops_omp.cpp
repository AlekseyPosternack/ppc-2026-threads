#include "sabutay_sparse_complex_ccs_mult_ompfix/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <complex>
#include <cstddef>

#include "sabutay_sparse_complex_ccs_mult_ompfix/common/include/common.hpp"

namespace sabutay_sparse_complex_ccs_mult_ompfix {

namespace {

constexpr double kEps = 1e-14;

auto IsValidStructure(const CCS &matrix) -> bool {
  if (matrix.m < 0 || matrix.n < 0) {
    return false;
  }
  if (matrix.col_ptr.size() != (static_cast<std::size_t>(matrix.n) + 1U)) {
    return false;
  }
  if (matrix.row_ind.size() != matrix.values.size()) {
    return false;
  }
  if (matrix.col_ptr.empty() || matrix.col_ptr.front() != 0) {
    return false;
  }
  if (matrix.col_ptr.back() != static_cast<int>(matrix.row_ind.size())) {
    return false;
  }

  for (int i = 0; i < matrix.n; ++i) {
    if (matrix.col_ptr[static_cast<std::size_t>(i)] > matrix.col_ptr[static_cast<std::size_t>(i + 1)]) {
      return false;
    }
  }

  for (const int row : matrix.row_ind) {
    if (row < 0 || row >= matrix.m) {
      return false;
    }
  }
  return true;
}

void BuildColumn(const CCS &a, const CCS &b, int col_index, std::vector<int> &marker,
                 std::vector<std::complex<double>> &acc, std::vector<int> &touched_rows,
                 std::vector<std::pair<int, std::complex<double>>> &result_column) {
  touched_rows.clear();

  for (int k = b.col_ptr[static_cast<std::size_t>(col_index)]; k < b.col_ptr[static_cast<std::size_t>(col_index + 1)];
       ++k) {
    const std::complex<double> b_value = b.values[static_cast<std::size_t>(k)];
    const int b_row = b.row_ind[static_cast<std::size_t>(k)];

    for (int az = a.col_ptr[static_cast<std::size_t>(b_row)]; az < a.col_ptr[static_cast<std::size_t>(b_row + 1)];
         ++az) {
      const int a_row = a.row_ind[static_cast<std::size_t>(az)];
      acc[static_cast<std::size_t>(a_row)] += b_value * a.values[static_cast<std::size_t>(az)];
      if (marker[static_cast<std::size_t>(a_row)] != col_index) {
        marker[static_cast<std::size_t>(a_row)] = col_index;
        touched_rows.push_back(a_row);
      }
    }
  }

  std::ranges::sort(touched_rows);
  result_column.clear();
  result_column.reserve(touched_rows.size());
  for (const int row : touched_rows) {
    const std::complex<double> value = acc[static_cast<std::size_t>(row)];
    if (std::abs(value) > kEps) {
      result_column.emplace_back(row, value);
    }
    acc[static_cast<std::size_t>(row)] = std::complex<double>(0.0, 0.0);
  }
}

}  // namespace

SabutaySparseComplexCcsMultOmpFix::SabutaySparseComplexCcsMultOmpFix(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CCS();
}

bool SabutaySparseComplexCcsMultOmpFix::ValidationImpl() {
  const CCS &a = std::get<0>(GetInput());
  const CCS &b = std::get<1>(GetInput());
  return a.n == b.m && IsValidStructure(a) && IsValidStructure(b);
}

bool SabutaySparseComplexCcsMultOmpFix::PreProcessingImpl() {
  GetOutput() = CCS();
  return true;
}

void SabutaySparseComplexCcsMultOmpFix::SpMM(const CCS &a, const CCS &b, CCS &c) {
  c.m = a.m;
  c.n = b.n;
  c.col_ptr.assign(b.n + 1, 0);
  c.row_ind.clear();
  c.values.clear();

  std::vector<std::vector<std::pair<int, std::complex<double>>>> columns(static_cast<std::size_t>(b.n));
  const int requested_threads = std::max(1, omp_get_max_threads());

#pragma omp parallel default(none) shared(a, b, columns) num_threads(requested_threads)
  {
    std::vector<int> marker(static_cast<std::size_t>(a.m), -1);
    std::vector<std::complex<double>> acc(static_cast<std::size_t>(a.m), std::complex<double>(0.0, 0.0));
    std::vector<int> touched_rows;

#pragma omp for schedule(static)
    for (int j = 0; j < b.n; ++j) {
      auto &column = columns[static_cast<std::size_t>(j)];
      BuildColumn(a, b, j, marker, acc, touched_rows, column);
    }
  }

  for (int j = 0; j < b.n; ++j) {
    const int col_size = static_cast<int>(columns[static_cast<std::size_t>(j)].size());
    c.col_ptr[static_cast<std::size_t>(j + 1)] = c.col_ptr[static_cast<std::size_t>(j)] + col_size;
  }

  const auto nnz = static_cast<std::size_t>(c.col_ptr.back());
  c.row_ind.resize(nnz);
  c.values.resize(nnz);

  for (int j = 0; j < b.n; ++j) {
    const int start = c.col_ptr[static_cast<std::size_t>(j)];
    const auto &column = columns[static_cast<std::size_t>(j)];
    for (std::size_t k = 0; k < column.size(); ++k) {
      const int dst = start + static_cast<int>(k);
      c.row_ind[static_cast<std::size_t>(dst)] = column[k].first;
      c.values[static_cast<std::size_t>(dst)] = column[k].second;
    }
  }
}

bool SabutaySparseComplexCcsMultOmpFix::RunImpl() {
  const CCS &a = std::get<0>(GetInput());
  const CCS &b = std::get<1>(GetInput());
  CCS &c = GetOutput();
  SpMM(a, b, c);
  return true;
}

bool SabutaySparseComplexCcsMultOmpFix::PostProcessingImpl() {
  return IsValidStructure(GetOutput());
}

}  // namespace sabutay_sparse_complex_ccs_mult_ompfix
