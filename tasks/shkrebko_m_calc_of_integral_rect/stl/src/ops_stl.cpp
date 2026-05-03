#include "shkrebko_m_calc_of_integral_rect/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "shkrebko_m_calc_of_integral_rect/common/include/common.hpp"
#include "util/include/util.hpp"

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
  if (!std::ranges::all_of(input.limits, [](const auto &lim) { return lim.first < lim.second; })) {
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

  // Вычисляем шаги h и объём одной ячейки
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

  // Количество потоков
  int num_threads = ppc::util::GetNumThreads();
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
  }
  if (num_threads <= 0) {
    num_threads = 2;
  }

  // Разбиваем диапазон [0, total_points) на блоки
  std::size_t chunk = total_points / num_threads;
  std::size_t remainder = total_points % num_threads;

  std::vector<std::thread> threads;
  std::vector<double> partial_sums(num_threads, 0.0);

  // Лямбда, вычисляющая сумму на отрезке [start, end)
  auto compute_chunk = [&](std::size_t start, std::size_t end, double &output) {
    thread_local std::vector<double> point;
    if (point.size() != dim) {
      point.resize(dim);
    }

    double local_sum = 0.0;
    for (std::size_t idx = start; idx < end; ++idx) {
      std::size_t tmp = idx;
      // Преобразование линейного индекса в многомерные координаты
      for (int i = static_cast<int>(dim) - 1; i >= 0; --i) {
        std::size_t coord_index = tmp % static_cast<std::size_t>(n_steps[i]);
        tmp /= static_cast<std::size_t>(n_steps[i]);
        point[i] = limits[i].first + (static_cast<double>(coord_index) + 0.5) * h[i];
      }
      local_sum += func(point);
    }
    output = local_sum;
  };

  std::size_t start = 0;
  for (int i = 0; i < num_threads; ++i) {
    std::size_t end = start + chunk + (static_cast<std::size_t>(i) < remainder ? 1 : 0);
    if (start >= end) {
      partial_sums[i] = 0.0;
      continue;
    }

    threads.emplace_back(compute_chunk, start, end, std::ref(partial_sums[i]));
    start = end;
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  double total_sum = 0.0;
  for (double s : partial_sums) {
    total_sum += s;
  }

  res_ = total_sum * cell_volume;
  return true;
}

bool ShkrebkoMCalcOfIntegralRectSTL::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace shkrebko_m_calc_of_integral_rect
