#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <stdexcept>
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

OutType RunCompletedTask(const InType& input) {
  DoubleSortEvenOddBatcherOMP task(input);
  if (!task.Validation()) {
    throw std::runtime_error("Validation failed");
  }
  if (!task.PreProcessing()) {
    throw std::runtime_error("PreProcessing failed");
  }
  if (!task.Run()) {
    throw std::runtime_error("Run failed");
  }
  if (!task.PostProcessing()) {
    throw std::runtime_error("PostProcessing failed");
  }

  return task.GetOutput();
}

void CheckMatchesStdSort(const InType& input) {
  auto expected = input;
  std::ranges::sort(expected);

  const auto output = RunCompletedTask(input);
  EXPECT_EQ(output, expected);
}

TEST(GusevDoubleSortEvenOddBatcherOMP, SortsEmptyInput) { CheckMatchesStdSort({}); }

TEST(GusevDoubleSortEvenOddBatcherOMP, SortsSingleElement) { CheckMatchesStdSort({42.0}); }

TEST(GusevDoubleSortEvenOddBatcherOMP, SortsOddSizedInput) { CheckMatchesStdSort({3.0, -1.0, 2.0, 0.0, 5.0}); }

TEST(GusevDoubleSortEvenOddBatcherOMP, MatchesStdSortForPrimeSizedRandomInput) {
  CheckMatchesStdSort(GenerateRandomInput(997, 20260320));
}

TEST(GusevDoubleSortEvenOddBatcherOMP, MatchesStdSortForExtremesAndDuplicates) {
  CheckMatchesStdSort({std::numbers::pi,
                       -std::numbers::e,
                       0.0,
                       -0.0,
                       42.0,
                       -42.0,
                       std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::lowest(),
                       std::numeric_limits<double>::min(),
                       -std::numeric_limits<double>::min(),
                       42.0,
                       -42.0,
                       7.5,
                       7.5,
                       -7.5});
}

TEST(GusevDoubleSortEvenOddBatcherOMP, ValidationRejectsPreparedOutput) {
  DoubleSortEvenOddBatcherOMP task({3.0, 2.0, 1.0});
  task.SetOutput({0.0});

  EXPECT_FALSE(task.Validation());
  EXPECT_THROW(task.Run(), std::runtime_error);
}

TEST(GusevDoubleSortEvenOddBatcherOMP, ThrowsIfPreProcessingBeforeValidation) {
  DoubleSortEvenOddBatcherOMP task({1.0});
  EXPECT_THROW(task.PreProcessing(), std::runtime_error);
}

TEST(GusevDoubleSortEvenOddBatcherOMP, ThrowsIfRunBeforePreProcessing) {
  DoubleSortEvenOddBatcherOMP task({2.0, 1.0});
  EXPECT_TRUE(task.Validation());
  EXPECT_THROW(task.Run(), std::runtime_error);
}

TEST(GusevDoubleSortEvenOddBatcherOMP, ThrowsIfPostProcessingBeforeRun) {
  DoubleSortEvenOddBatcherOMP task({2.0, 1.0});
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_THROW(task.PostProcessing(), std::runtime_error);
}

TEST(GusevDoubleSortEvenOddBatcherOMP, AllowsRepeatedRunBeforePostProcessing) {
  const InType input{9.0, -1.0, 5.0, 3.0, -7.0, 11.0, 0.0, 2.0};
  const auto reference_output = RunCompletedTask(input);

  DoubleSortEvenOddBatcherOMP task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), reference_output);
}

TEST(GusevDoubleSortEvenOddBatcherOMP, UsesInputSnapshotFromPreProcessing) {
  DoubleSortEvenOddBatcherOMP task({5.0, 4.0, 3.0, 2.0, 1.0});

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());

  auto& input_ref = task.GetInput();
  input_ref[0] = -100.0;
  input_ref[1] = 200.0;

  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), (OutType{1.0, 2.0, 3.0, 4.0, 5.0}));
}

}  // namespace
