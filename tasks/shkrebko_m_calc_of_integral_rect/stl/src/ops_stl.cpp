#include "shkrebko_m_calc_of_integral_rect/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <numeric>
#include <vector>

#include "shkrebko_m_calc_of_integral_rect/common/include/common.hpp"

namespace shkrebko_m_calc_of_integral_rect {

ShkrebkoMCalcOfIntegralRectSTL::ShkrebkoMCalcOfIntegralRectSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
}

bool ShkrebkoMCalcOfIntegralRectSTL::ValidationImpl() {
  const auto &input = GetInput();

  if (!input.func) {
    return false;
  }
  if (input.limits.size() != input.n_steps.size() || input.limits.empty()) {
    return false;
  }
  if (!std::ranges::all_of(input.n_steps, [](int n) { return n > 0; })) {
    return false;
  }
  if (!std::ranges::all_of(input.limits,
                           [](const auto &lim) { return lim.first < lim.second; })) {
    return false;
  }
  return true;
}

bool ShkrebkoMCalcOfIntegralRectSTL::PreProcessingImpl() {
  local_input_ = GetInput();
  res_ = 0.0;
  return true;
}

bool ShkrebkoMCalcOfIntegralRectSTL::RunImpl() {
  const std::size_t dim = local_input_.limits.size();
  const auto &limits = local_input_.limits;
  const auto &n_steps = local_input_.n_steps;
  const auto &func = local_input_.func;

  std::vector<double> h(dim);
  double cell_volume = 1.0;
  std::size_t total_points = 1;
  for (std::size_t i = 0; i < dim; ++i) {
    double left = limits[i].first;
    double right = limits[i].second;
    int steps = n_steps[i];
    h[i] = (right - left) / static_cast<double>(steps);
    cell_volume *= h[i];
    total_points *= static_cast<std::size_t>(steps);
  }

  std::vector<std::size_t> indices(total_points);
  std::iota(indices.begin(), indices.end(), 0);

  double total_sum = std::transform_reduce(
      std::execution::par,
      indices.begin(),
      indices.end(),
      0.0,
      std::plus<>(),
      [&](std::size_t idx) -> double {
        thread_local std::vector<double> point;
        thread_local std::size_t prev_dim = 0;

        if (point.size() != dim) {
          point.resize(dim);
          prev_dim = dim;
        }

        std::size_t tmp = idx;
        for (int i = static_cast<int>(dim) - 1; i >= 0; --i) {
          std::size_t coord_index = tmp % static_cast<std::size_t>(n_steps[i]);
          tmp /= static_cast<std::size_t>(n_steps[i]);
          point[i] = limits[i].first + (static_cast<double>(coord_index) + 0.5) * h[i];
        }

        return func(point);
      });

  res_ = total_sum * cell_volume;
  return true;
}

bool ShkrebkoMCalcOfIntegralRectSTL::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace shkrebko_m_calc_of_integral_rect