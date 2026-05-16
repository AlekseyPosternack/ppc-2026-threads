#include "posternak_a_crs_mul_complex_matrix/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "posternak_a_crs_mul_complex_matrix/common/include/common.hpp"

namespace {

size_t ComputeRowNoZeroCount(const posternak_a_crs_mul_complex_matrix::CRSMatrix &a,
                             const posternak_a_crs_mul_complex_matrix::CRSMatrix &b, int row, double threshold) {
  std::unordered_map<int, std::complex<double>> row_sum;

  for (int idx_a = a.index_row[row]; idx_a < a.index_row[row + 1]; ++idx_a) {
    int col_a = a.index_col[idx_a];
    auto val_a = a.values[idx_a];

    for (int idx_b = b.index_row[col_a]; idx_b < b.index_row[col_a + 1]; ++idx_b) {
      int col_b = b.index_col[idx_b];
      auto val_b = b.values[idx_b];
      row_sum[col_b] += val_a * val_b;
    }
  }

  size_t local = 0;
  for (const auto &[col, val] : row_sum) {
    if (std::abs(val) > threshold) {
      ++local;
    }
  }
  return local;
}

void BuildResultStructure(posternak_a_crs_mul_complex_matrix::CRSMatrix &res, std::vector<size_t> &row_prefix) {
  for (int i = 1; i < res.rows; ++i) {
    row_prefix[i] += row_prefix[i - 1];
  }

  const size_t total = row_prefix.empty() ? 0 : row_prefix.back();
  res.values.resize(total);
  res.index_col.resize(total);
  res.index_row.resize(res.rows + 1);

  for (int i = 0; i <= res.rows; ++i) {
    res.index_row[i] = (i == 0 ? 0 : static_cast<int>(row_prefix[i - 1]));
  }
}

void ComputeAndWriteRow(const posternak_a_crs_mul_complex_matrix::CRSMatrix &a,
                        const posternak_a_crs_mul_complex_matrix::CRSMatrix &b,
                        posternak_a_crs_mul_complex_matrix::CRSMatrix &res, int row, double threshold) {
  std::unordered_map<int, std::complex<double>> row_sum;

  for (int idx_a = a.index_row[row]; idx_a < a.index_row[row + 1]; ++idx_a) {
    int col_a = a.index_col[idx_a];
    auto val_a = a.values[idx_a];

    for (int idx_b = b.index_row[col_a]; idx_b < b.index_row[col_a + 1]; ++idx_b) {
      int col_b = b.index_col[idx_b];
      auto val_b = b.values[idx_b];
      row_sum[col_b] += val_a * val_b;
    }
  }

  std::vector<std::pair<int, std::complex<double>>> sorted(row_sum.begin(), row_sum.end());
  std::ranges::sort(sorted, [](const auto &p1, const auto &p2) { return p1.first < p2.first; });

  size_t pos = res.index_row[row];
  for (const auto &[col_idx, value] : sorted) {
    if (std::abs(value) > threshold) {
      res.values[pos] = value;
      res.index_col[pos] = col_idx;
      ++pos;
    }
  }
}

struct RowDistribution {
  int local_start;
  int local_end;
  int local_count;
  int rows_per_proc;
  int rem;
};

RowDistribution CalculateRowDistribution(int total_rows, int rank, int size) {
  int rows_per_proc = total_rows / size;
  int rem = total_rows % size;
  int local_start = rank * rows_per_proc + std::min(rank, rem);
  int local_end = local_start + rows_per_proc + (rank < rem ? 1 : 0);
  return {local_start, local_end, local_end - local_start, rows_per_proc, rem};
}

std::vector<size_t> ComputeLocalRowCounts(const posternak_a_crs_mul_complex_matrix::CRSMatrix &a,
                                          const posternak_a_crs_mul_complex_matrix::CRSMatrix &b, int local_start,
                                          int local_count, double threshold) {
  std::vector<size_t> local_counts(local_count);
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < local_count; ++i) {
    local_counts[i] = ComputeRowNoZeroCount(a, b, local_start + i, threshold);
  }
  return local_counts;
}

std::vector<size_t> GatherRowCountsToRoot(const std::vector<size_t> &local_counts, const RowDistribution &dist,
                                          int total_rows, int rank, int size) {
  std::vector<int> recv_counts(size), displs(size);
  for (int p = 0; p < size; ++p) {
    int p_start = p * dist.rows_per_proc + std::min(p, dist.rem);
    int p_end = p_start + dist.rows_per_proc + (p < dist.rem ? 1 : 0);
    recv_counts[p] = p_end - p_start;
    displs[p] = p_start;
  }

  std::vector<size_t> global_counts(total_rows);
  if (rank == 0) {
    MPI_Gatherv(local_counts.data(), static_cast<int>(local_counts.size()), MPI_UNSIGNED_LONG, global_counts.data(),
                recv_counts.data(), displs.data(), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_counts.data(), static_cast<int>(local_counts.size()), MPI_UNSIGNED_LONG, nullptr, nullptr,
                nullptr, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  }
  return global_counts;
}

void BroadcastResultStructure(posternak_a_crs_mul_complex_matrix::CRSMatrix &res, std::vector<size_t> &global_counts,
                              int rank) {
  if (rank == 0) {
    BuildResultStructure(res, global_counts);
  }
  res.index_row.resize(res.rows + 1);
  MPI_Bcast(res.index_row.data(), static_cast<int>(res.index_row.size()), MPI_INT, 0, MPI_COMM_WORLD);
}

void ComputeLocalRows(const posternak_a_crs_mul_complex_matrix::CRSMatrix &a,
                      const posternak_a_crs_mul_complex_matrix::CRSMatrix &b,
                      posternak_a_crs_mul_complex_matrix::CRSMatrix &res, int local_start, int local_count,
                      double threshold) {
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < local_count; ++i) {
    ComputeAndWriteRow(a, b, res, local_start + i, threshold);
  }
}

struct GatherParams {
  std::vector<int> counts;
  std::vector<int> displs;
};

GatherParams PrepareGatherParams(const posternak_a_crs_mul_complex_matrix::CRSMatrix &res, const RowDistribution &dist,
                                 int size) {
  std::vector<int> g_counts(size), g_displs(size);
  for (int p = 0; p < size; ++p) {
    int p_start = p * dist.rows_per_proc + std::min(p, dist.rem);
    int p_end = p_start + dist.rows_per_proc + (p < dist.rem ? 1 : 0);
    g_displs[p] = res.index_row[p_start];
    g_counts[p] = res.index_row[p_end] - res.index_row[p_start];
  }
  return {std::move(g_counts), std::move(g_displs)};
}

void GatherResultData(const posternak_a_crs_mul_complex_matrix::CRSMatrix &res, int local_start, int local_end,
                      const GatherParams &params, int rank, int size) {
  int local_nnz = res.index_row[local_end] - res.index_row[local_start];

  MPI_Gatherv(res.values.data() + res.index_row[local_start], local_nnz, MPI_C_DOUBLE_COMPLEX,
              rank == 0 ? res.values.data() : nullptr, params.counts.data(), params.displs.data(), MPI_C_DOUBLE_COMPLEX,
              0, MPI_COMM_WORLD);
  MPI_Gatherv(res.index_col.data() + res.index_row[local_start], local_nnz, MPI_INT,
              rank == 0 ? res.index_col.data() : nullptr, params.counts.data(), params.displs.data(), MPI_INT, 0,
              MPI_COMM_WORLD);
}

void BroadcastResultData(posternak_a_crs_mul_complex_matrix::CRSMatrix &res, int rank) {
  int total_nnz = res.index_row.back();
  MPI_Bcast(res.values.data(), total_nnz, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  MPI_Bcast(res.index_col.data(), total_nnz, MPI_INT, 0, MPI_COMM_WORLD);
}

}  // namespace

namespace posternak_a_crs_mul_complex_matrix {

PosternakACRSMulComplexMatrixALL::PosternakACRSMulComplexMatrixALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = CRSMatrix{};
}

bool PosternakACRSMulComplexMatrixALL::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = input.first;
  const auto &b = input.second;
  return a.IsValid() && b.IsValid() && a.cols == b.rows;
}

bool PosternakACRSMulComplexMatrixALL::PreProcessingImpl() {
  const auto &input = GetInput();
  const auto &a = input.first;
  const auto &b = input.second;
  auto &res = GetOutput();

  res.rows = a.rows;
  res.cols = b.cols;
  return true;
}

bool PosternakACRSMulComplexMatrixALL::RunImpl() {
  const auto &input = GetInput();
  const auto &a = input.first;
  const auto &b = input.second;
  auto &res = GetOutput();

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Обработка пустых матриц
  if (a.values.empty() || b.values.empty()) {
    res.rows = a.rows;
    res.cols = b.cols;
    res.values.clear();
    res.index_col.clear();
    res.index_row.assign(res.rows + 1, 0);
    return true;
  }

  constexpr double kThreshold = 1e-12;

  // Распределение строк
  auto dist = CalculateRowDistribution(res.rows, rank, size);

  // подсчет количества ненулевых элементов для локальных строк
  auto local_counts = ComputeLocalRowCounts(a, b, dist.local_start, dist.local_count, kThreshold);

  // структура результата
  auto global_counts = GatherRowCountsToRoot(local_counts, dist, res.rows, rank, size);
  BroadcastResultStructure(res, global_counts, rank);

  // вычисление значений для локальных строк
  ComputeLocalRows(a, b, res, dist.local_start, dist.local_count, kThreshold);

  // сбор данных
  auto gather_params = PrepareGatherParams(res, dist, size);
  GatherResultData(res, dist.local_start, dist.local_end, gather_params, rank, size);

  BroadcastResultData(res, rank);

  return res.IsValid();
}

bool PosternakACRSMulComplexMatrixALL::PostProcessingImpl() {
  return GetOutput().IsValid();
}

}  // namespace posternak_a_crs_mul_complex_matrix
