#include "frolova_s_radix_sort_double/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <vector>

#include "frolova_s_radix_sort_double/common/include/common.hpp"

namespace frolova_s_radix_sort_double {

FrolovaSRadixSortDoubleTBB::FrolovaSRadixSortDoubleTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool FrolovaSRadixSortDoubleTBB::ValidationImpl() {
  return !GetInput().empty();
}

bool FrolovaSRadixSortDoubleTBB::PreProcessingImpl() {
  return true;
}

bool FrolovaSRadixSortDoubleTBB::RunImpl() {
  const std::vector<double> &input = GetInput();
  if (input.empty()) {
    return false;
  }

  std::vector<double> working = input;
  const std::size_t n = working.size();

  constexpr int kRadix = 256;
  constexpr int kBits = 8;
  constexpr int kPasses = sizeof(std::uint64_t);

  std::vector<double> temp(n);

  for (int pass = 0; pass < kPasses; ++pass) {
    std::array<std::atomic<int>, kRadix> count{};

    // parallel histogram
    tbb::parallel_for(std::size_t(0), n, [&](std::size_t i) {
      auto bits = std::bit_cast<std::uint64_t>(working[i]);
      int byte = static_cast<int>((bits >> (pass * kBits)) & 0xFF);
      count[byte].fetch_add(1, std::memory_order_relaxed);
    });

    // compute offsets (prefix sum)
    std::array<std::atomic<int>, kRadix> offset;
    int total = 0;
    for (int i = 0; i < kRadix; ++i) {
      int c = count[i].load();
      offset[i].store(total);
      total += c;
    }

    // distribute with atomic offsets
    tbb::parallel_for(std::size_t(0), n, [&](std::size_t i) {
      auto bits = std::bit_cast<std::uint64_t>(working[i]);
      int byte = static_cast<int>((bits >> (pass * kBits)) & 0xFF);
      int idx = offset[byte].fetch_add(1);
      temp[idx] = working[i];
    });

    working.swap(temp);
  }

  // post-process for IEEE 754 double: separate and reverse negatives
  std::vector<double> negative;
  std::vector<double> positive;
  negative.reserve(n);
  positive.reserve(n);

  for (double val : working) {
    if (val < 0.0) {
      negative.push_back(val);
    } else {
      positive.push_back(val);
    }
  }
  std::reverse(negative.begin(), negative.end());

  working.clear();
  working.insert(working.end(), negative.begin(), negative.end());
  working.insert(working.end(), positive.begin(), positive.end());

  GetOutput() = std::move(working);
  return true;
}

bool FrolovaSRadixSortDoubleTBB::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_radix_sort_double
