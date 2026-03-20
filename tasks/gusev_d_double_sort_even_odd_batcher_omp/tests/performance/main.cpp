#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <ranges>
#include <vector>

#include "tasks/gusev_d_double_sort_even_odd_batcher_omp/omp/include/ops_omp.hpp"

namespace {

using namespace gusev_d_double_sort_even_odd_batcher_omp_task_threads;

InType GenerateRandomInput(size_t size, uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<double> distribution(-1.0e6, 1.0e6);

  InType input(size);
  for (double& value : input) {
    value = distribution(generator);
  }

  return input;
}

InType GenerateDescendingInput(size_t size) {
  InType input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<double>(size - i);
  }

  return input;
}

InType GenerateNearlySortedInput(size_t size) {
  InType input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<double>(i);
  }

  for (size_t i = 1; i < size; i += 64) {
    std::swap(input[i - 1], input[i]);
  }

  return input;
}

void RunPerfCase(const InType& input) {
  auto expected = input;
  std::ranges::sort(expected);

  DoubleSortEvenOddBatcherOMP task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());

  const auto started = std::chrono::steady_clock::now();
  ASSERT_TRUE(task.Run());
  const auto finished = std::chrono::steady_clock::now();

  ASSERT_TRUE(task.PostProcessing());
  EXPECT_EQ(task.GetOutput(), expected);

  const std::chrono::duration<double> elapsed = finished - started;
  std::cout << "omp_run_time_sec:" << elapsed.count() << '\n';
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPDescending) {
  RunPerfCase(GenerateDescendingInput(1 << 15));
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPRandom) {
  RunPerfCase(GenerateRandomInput(1 << 15, 20260320));
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPNearlySorted) {
  RunPerfCase(GenerateNearlySortedInput(1 << 15));
}

}  // namespace
