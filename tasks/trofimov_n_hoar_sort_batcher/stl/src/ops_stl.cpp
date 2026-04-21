#include "trofimov_n_hoar_sort_batcher/stl/include/ops_stl.hpp"

#include <algorithm>
#include <thread>
#include <vector>

#include "trofimov_n_hoar_sort_batcher/common/include/common.hpp"

namespace trofimov_n_hoar_sort_batcher {

namespace {

int HoarePartition(std::vector<int> &arr, int left, int right) {
  const int pivot = arr[left + ((right - left) / 2)];
  int i = left - 1;
  int j = right + 1;

  while (true) {
    while (arr[++i] < pivot) {
    }

    while (arr[--j] > pivot) {
    }

    if (i >= j) {
      return j;
    }

    std::swap(arr[i], arr[j]);
  }
}

void QuickSortStlTask(std::vector<int> &arr, int left, int right, int depth_limit) {
  constexpr int kSequentialThreshold = 1024;

  struct Segment {
    int left;
    int right;
    int depth;
  };

  std::vector<Segment> stack;
  stack.push_back({left, right, depth_limit});

  while (!stack.empty()) {
    Segment seg = stack.back();
    stack.pop_back();

    if (seg.left >= seg.right) {
      continue;
    }

    if ((seg.right - seg.left) < kSequentialThreshold || seg.depth <= 0) {
      std::sort(arr.begin() + seg.left, arr.begin() + seg.right + 1);
      continue;
    }

    const int split = HoarePartition(arr, seg.left, seg.right);

    stack.push_back({split + 1, seg.right, seg.depth - 1});
    stack.push_back({seg.left, split, seg.depth - 1});
  }
}

}  // namespace

TrofimovNHoarSortBatcherSTL::TrofimovNHoarSortBatcherSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool TrofimovNHoarSortBatcherSTL::ValidationImpl() {
  return true;
}

bool TrofimovNHoarSortBatcherSTL::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool TrofimovNHoarSortBatcherSTL::RunImpl() {
  auto &data = GetOutput();

  if (data.size() > 1) {
    QuickSortStlTask(data, 0, static_cast<int>(data.size()) - 1, 4);
  }

  return true;
}

bool TrofimovNHoarSortBatcherSTL::PostProcessingImpl() {
  return true;
}

}  // namespace trofimov_n_hoar_sort_batcher
