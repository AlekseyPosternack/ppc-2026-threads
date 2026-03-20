#pragma once

#include <vector>

#include "modules/task/include/task.hpp"

namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads {

using InType = std::vector<double>;
using OutType = std::vector<double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads
