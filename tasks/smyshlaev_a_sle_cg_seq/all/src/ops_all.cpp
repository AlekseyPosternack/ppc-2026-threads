#include "smyshlaev_a_sle_cg_seq/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>
#include <algorithm>

#include "smyshlaev_a_sle_cg_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace smyshlaev_a_sle_cg_seq {

SmyshlaevASleCgTaskALL::SmyshlaevASleCgTaskALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  if (ppc::util::IsUnderMpirun()) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
}

bool SmyshlaevASleCgTaskALL::ValidationImpl() {
  int rank = 0;
  if (ppc::util::IsUnderMpirun()) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int error = 0;
  if (rank == 0) {
    const auto &a = GetInput().A;
    const auto &b = GetInput().b;
    if (a.empty() || b.empty() || a.size() != b.size() || a.size() != a[0].size()) {
      error = 1;
    }
  }
  if (ppc::util::IsUnderMpirun()) MPI_Bcast(&error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return error == 0;
}

bool SmyshlaevASleCgTaskALL::PreProcessingImpl() {
  int rank = 0;
  if (ppc::util::IsUnderMpirun()) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &a = GetInput().A;
    n_ = static_cast<int>(a.size());
    flat_A_.resize(static_cast<size_t>(n_) * n_);
    for (int i = 0; i < n_; ++i) {
      for (int j = 0; j < n_; ++j) {
        flat_A_[(static_cast<size_t>(i) * n_) + j] = a[i][j];
      }
    }
  }
  return true;
}

bool SmyshlaevASleCgTaskALL::RunImpl() {
  int rank = 0;
  int size = 1;
  bool is_mpi = ppc::util::IsUnderMpirun();
  if (is_mpi) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  if (is_mpi) MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (is_mpi && rank != 0) {
    flat_A_.resize(static_cast<size_t>(n_) * n_);
  }
  std::vector<double> b_vector(n_);
  if (rank == 0) b_vector = GetInput().b;

  if (is_mpi) {
    MPI_Bcast(flat_A_.data(), static_cast<int>(flat_A_.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b_vector.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  int proc_chunk = n_ / size;
  int proc_remainder = n_ % size;
  int my_start = (rank * proc_chunk) + std::min(rank, proc_remainder);
  int my_end = my_start + proc_chunk + (rank < proc_remainder ? 1 : 0);
  int my_count = my_end - my_start;

  std::vector<double> r = b_vector; 
  std::vector<double> p = r;        
  std::vector<double> x(n_, 0.0);   
  std::vector<double> ap(n_, 0.0);

  int num_threads = ppc::util::GetNumThreads();
  omp_set_num_threads(num_threads);

  double local_rs_old = 0.0;
  #pragma omp parallel for reduction(+ : local_rs_old)
  for (int i = my_start; i < my_end; ++i) {
    local_rs_old += r[i] * r[i];
  }

  double rs_old = 0.0;
  if (is_mpi) {
    MPI_Allreduce(&local_rs_old, &rs_old, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  } else {
    rs_old = local_rs_old;
  }

  const double epsilon = 1e-9;
  for (int iter = 0; iter < n_ * 2; ++iter) {
    if (std::sqrt(rs_old) < epsilon) break;

    #pragma omp parallel for
    for (int i = my_start; i < my_end; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n_; ++j) {
        sum += flat_A_[(static_cast<size_t>(i) * n_) + j] * p[j];
      }
      ap[i] = sum;
    }

    double local_p_ap = 0.0;
    #pragma omp parallel for reduction(+ : local_p_ap)
    for (int i = my_start; i < my_end; ++i) {
      local_p_ap += p[i] * ap[i];
    }
    double p_ap = 0.0;
    if (is_mpi) MPI_Allreduce(&local_p_ap, &p_ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    else p_ap = local_p_ap;

    if (std::abs(p_ap) < 1e-15) break;

    double alpha = rs_old / p_ap;

    double local_rs_new = 0.0;
    #pragma omp parallel for reduction(+ : local_rs_new)
    for (int i = my_start; i < my_end; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
      local_rs_new += r[i] * r[i];
    }
    double rs_new = 0.0;
    if (is_mpi) MPI_Allreduce(&local_rs_new, &rs_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    else rs_new = local_rs_new;

    if (std::sqrt(rs_new) < epsilon) break;

    double beta = rs_new / rs_old;

    #pragma omp parallel for
    for (int i = my_start; i < my_end; ++i) {
      p[i] = r[i] + (beta * p[i]);
    }

    if (is_mpi) {
      std::vector<int> counts(size);
      std::vector<int> displs(size);
      for (int i = 0; i < size; ++i) {
        counts[i] = (n_ / size) + (i < (n_ % size) ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
      }
      MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, p.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    }
    rs_old = rs_new;
  }

  if (is_mpi) {
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
      counts[i] = (n_ / size) + (i < (n_ % size) ? 1 : 0);
      displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
    }
    if (rank == 0) {
      MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
      res_ = x;
    } else {
      MPI_Gatherv(x.data() + my_start, my_count, MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
  } else {
    res_ = x;
  }

  return true;
}

bool SmyshlaevASleCgTaskALL::PostProcessingImpl() {
  int rank = 0;
  int is_mpi = ppc::util::IsUnderMpirun();
  if (is_mpi) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (is_mpi) {
    MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) res_.resize(n_);
    MPI_Bcast(res_.data(), n_, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  
  GetOutput() = res_;
  return true;
}

}  // namespace smyshlaev_a_sle_cg_seq