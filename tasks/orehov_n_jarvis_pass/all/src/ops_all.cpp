#include "orehov_n_jarvis_pass/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
#include <set>
#include <vector>

#include "orehov_n_jarvis_pass/common/include/common.hpp"

namespace orehov_n_jarvis_pass {

bool OrehovNJarvisPassALL::IsBetterPoint(const Point& current, const Point& candidate, const Point& best) {
  double orient = CheckLeft(current, best, candidate);
  if (orient > 0.0) return true;
  if (orient == 0.0) {
    double dx_c = candidate.x - current.x;
    double dy_c = candidate.y - current.y;
    double dist_c = dx_c * dx_c + dy_c * dy_c;

    double dx_b = best.x - current.x;
    double dy_b = best.y - current.y;
    double dist_b = dx_b * dx_b + dy_b * dy_b;

    return dist_c > dist_b;
  }
  return false;
}

OrehovNJarvisPassALL::BestState OrehovNJarvisPassALL::ReduceBestStates(const BestState& a, const BestState& b,
                                                                        const Point& current) {
  if (!a.valid) return b;
  if (!b.valid) return a;
  return BestState{.point = IsBetterPoint(current, b.point, a.point) ? b.point : a.point, .valid = true};
}

OrehovNJarvisPassALL::OrehovNJarvisPassALL(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<Point>();
}

bool OrehovNJarvisPassALL::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    return !GetInput().empty();
  }
  return true;
}

bool OrehovNJarvisPassALL::PreProcessingImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::set<Point> tmp(GetInput().begin(), GetInput().end());
    input_.assign(tmp.begin(), tmp.end());
  }

  size_t vec_size = (rank == 0) ? input_.size() : 0;
  MPI_Bcast(&vec_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    input_.resize(vec_size);
  }

  std::vector<double> buffer(vec_size * 2);
  if (rank == 0) {
    for (size_t i = 0; i < vec_size; ++i) {
      buffer[2 * i]     = input_[i].x;
      buffer[2 * i + 1] = input_[i].y;
    }
  }
  MPI_Bcast(buffer.data(), static_cast<int>(vec_size * 2), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    for (size_t i = 0; i < vec_size; ++i) {
      input_[i].x = buffer[2 * i];
      input_[i].y = buffer[2 * i + 1];
    }
  }
  return true;
}

bool OrehovNJarvisPassALL::RunImpl() {
  if (input_.size() == 1 || input_.size() == 2) {
    res_ = input_;
    return true;
  }

  Point current = FindFirstElem();
  res_.push_back(current);

  while (true) {
    Point next = FindNext(current);
    if (next == res_[0]) break;
    current = next;
    res_.push_back(next);
  }
  return true;
}

OrehovNJarvisPassALL::BestState OrehovNJarvisPassALL::LocalFindBest(const Point& current, size_t start,
                                                                     size_t end) const {
  BestState local_best;
  #pragma omp parallel
  {
    BestState thread_best;
    #pragma omp for nowait
    for (size_t i = start; i < end; ++i) {
      const Point& p = input_[i];
      if (p == current) continue;
      if (!thread_best.valid || IsBetterPoint(current, p, thread_best.point)) {
        thread_best.point = p;
        thread_best.valid = true;
      }
    }
    #pragma omp critical
    {
      if (thread_best.valid) {
        if (!local_best.valid) {
          local_best = thread_best;
        } else {
          local_best = ReduceBestStates(local_best, thread_best, current);
        }
      }
    }
  }
  return local_best;
}

OrehovNJarvisPassALL::BestState OrehovNJarvisPassALL::GlobalReduce(const std::vector<double>& all_data, int size,
                                                                     const Point& current) const {
  BestState global_best;
  for (int i = 0; i < size; ++i) {
    double x = all_data[3 * i];
    double y = all_data[3 * i + 1];
    bool v   = (all_data[3 * i + 2] != 0.0);
    if (v) {
      BestState proc_best{Point(x, y), true};
      if (!global_best.valid)
        global_best = proc_best;
      else
        global_best = ReduceBestStates(global_best, proc_best, current);
    }
  }
  return global_best;
}

OrehovNJarvisPassALL::BestState OrehovNJarvisPassALL::FinalizeBestPoint(const double* global_data) const {
  return BestState{Point(global_data[0], global_data[1]), (global_data[2] != 0.0)};
}

Point OrehovNJarvisPassALL::FindNext(Point current) const {
  const size_t n = input_.size();
  if (n == 0) return current;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t chunk = n / size;
  size_t rest  = n % size;
  size_t start = rank * chunk + std::min(static_cast<size_t>(rank), rest);
  size_t end   = start + chunk + (rank < static_cast<int>(rest) ? 1 : 0);

  BestState local_best = LocalFindBest(current, start, end);

  double local_data[3] = {local_best.valid ? local_best.point.x : 0.0,
                          local_best.valid ? local_best.point.y : 0.0,
                          local_best.valid ? 1.0 : 0.0};

  std::vector<double> all_data(size * 3);
  MPI_Gather(local_data, 3, MPI_DOUBLE, all_data.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double global_data[3] = {0.0, 0.0, 0.0};
  if (rank == 0) {
    BestState global_best = GlobalReduce(all_data, size, current);
    if (global_best.valid) {
      global_data[0] = global_best.point.x;
      global_data[1] = global_best.point.y;
      global_data[2] = 1.0;
    }
  }

  MPI_Bcast(global_data, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  BestState final_best = FinalizeBestPoint(global_data);
  return final_best.valid ? final_best.point : current;
}

double OrehovNJarvisPassALL::CheckLeft(Point a, Point b, Point c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

Point OrehovNJarvisPassALL::FindFirstElem() const {
  Point current = input_[0];
  for (const auto& f : input_) {
    if (f.x < current.x || (f.y < current.y && f.x == current.x)) {
      current = f;
    }
  }
  return current;
}

bool OrehovNJarvisPassALL::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace orehov_n_jarvis_pass