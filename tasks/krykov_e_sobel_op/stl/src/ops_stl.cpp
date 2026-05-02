#include "krykov_e_sobel_op/stl/include/ops_stl.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "krykov_e_sobel_op/common/include/common.hpp"

namespace krykov_e_sobel_op {

KrykovESobelOpSTL::KrykovESobelOpSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KrykovESobelOpSTL::ValidationImpl() {
  const auto &img = GetInput();
  return img.width > 2 && img.height > 2 && static_cast<int>(img.data.size()) == img.width * img.height;
}

bool KrykovESobelOpSTL::PreProcessingImpl() {
  const auto &img = GetInput();

  width_ = img.width;
  height_ = img.height;

  grayscale_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_));
  // RGB → grayscale
  for (int i = 0; i < width_ * height_; ++i) {
    const Pixel &p = img.data[i];
    grayscale_[i] = static_cast<int>((0.299 * p.r) + (0.587 * p.g) + (0.114 * p.b));
  }
  GetOutput().assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 0);
  return true;
}

bool KrykovESobelOpSTL::RunImpl() {
  const std::array<std::array<int, 3>, 3> gx_kernel = {{{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}}};
  const std::array<std::array<int, 3>, 3> gy_kernel = {{{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}}};

  auto &output = GetOutput();
  const auto &gray = grayscale_;
  const int h = height_;
  const int w = width_;


  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2; 
  }

  const int total_rows = h - 2;  
  if (total_rows <= 0) {
    return true;  
  }


  const int rows_per_thread = total_rows / static_cast<int>(num_threads);
  const int remainder = total_rows % static_cast<int>(num_threads);

  int next_start = 1;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (unsigned int i = 0; i < num_threads; ++i) {
    int chunk_rows = rows_per_thread + (i < static_cast<unsigned>(remainder) ? 1 : 0);
    int start_row = next_start;
    int end_row = start_row + chunk_rows;
    next_start = end_row;

    threads.emplace_back(
        [&, start_row, end_row]() {
          for (int row = start_row; row < end_row; ++row) {
            for (int col = 1; col < w - 1; ++col) {
              int gx = 0;
              int gy = 0;

              for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                  int pixel = gray[((row + ky) * w) + (col + kx)];
                  gx += pixel * gx_kernel[ky + 1][kx + 1];
                  gy += pixel * gy_kernel[ky + 1][kx + 1];
                }
              }

              int magnitude = static_cast<int>(
                  std::sqrt(static_cast<double>((gx * gx) + (gy * gy))));
              output[(row * w) + col] = magnitude;
            }
          }
        });
  }

  for (auto &t : threads) {
    t.join();
  }

  return true;
}

bool KrykovESobelOpSTL::PostProcessingImpl() {
  return true;
}

}  // namespace krykov_e_sobel_op
