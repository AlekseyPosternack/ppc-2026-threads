#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace sabutay_sparse_complex_ccs_mult_stl {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace sabutay_sparse_complex_ccs_mult_stl
