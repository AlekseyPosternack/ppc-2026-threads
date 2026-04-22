#include "kolotukhin_a_gaussian_blur/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include "kolotukhin_a_gaussian_blur/common/include/common.hpp"

namespace kolotukhin_a_gaussian_blur {

KolotukhinAGaussinBlureTBB::KolotukhinAGaussinBlureTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KolotukhinAGaussinBlureTBB::ValidationImpl() {
  const auto &pixel_data = std::get<0>(GetInput());
  const auto img_width = std::get<1>(GetInput());
  const auto img_height = std::get<2>(GetInput());

  return static_cast<std::size_t>(img_height) * static_cast<std::size_t>(img_width) == pixel_data.size();
}

bool KolotukhinAGaussinBlureTBB::PreProcessingImpl() {
  const auto img_width = std::get<1>(GetInput());
  const auto img_height = std::get<2>(GetInput());

  GetOutput().assign(static_cast<std::size_t>(img_height) * static_cast<std::size_t>(img_width), 0);
  return true;
}

bool KolotukhinAGaussinBlureTBB::RunImpl() {
  const auto &pixel_data = get<0>(GetInput());
  const auto img_width = get<1>(GetInput());
  const auto img_height = get<2>(GetInput());

  auto &output = GetOutput();

  int expanded_width = img_width + 2;
  int expanded_height = img_height + 2;
  std::vector<std::uint8_t> expanded_pixel_data(expanded_width * expanded_height);

  tbb::parallel_for(0, img_height, [&](int y) {
    size_t src_row = y * img_width;
    size_t dst_row = (y + 1) * expanded_width + 1;
    std::copy(pixel_data.begin() + src_row, pixel_data.begin() + src_row + img_width,
              expanded_pixel_data.begin() + dst_row);

    expanded_pixel_data[dst_row - 1] = expanded_pixel_data[dst_row];
    expanded_pixel_data[dst_row + img_width] = expanded_pixel_data[dst_row + img_width - 1];
  });

  std::copy(expanded_pixel_data.begin() + expanded_width, expanded_pixel_data.begin() + 2 * expanded_width,
            expanded_pixel_data.begin());
  std::copy(expanded_pixel_data.end() - 2 * expanded_width, expanded_pixel_data.end() - expanded_width,
            expanded_pixel_data.end() - expanded_width);

  std::vector<std::uint8_t> temp(expanded_pixel_data.size());

  tbb::parallel_for(0, expanded_height, [&](int y) {
    const uint8_t *row = expanded_pixel_data.data() + y * expanded_width;
    uint8_t *out_row = temp.data() + y * expanded_width;

    for (int x = 1; x < expanded_width - 1; x++) {
      int sum = row[x - 1] + 2 * row[x] + row[x + 1];
      out_row[x] = static_cast<uint8_t>(sum / 4);
    }
  });

  tbb::parallel_for(1, expanded_height - 1, [&](int y) {
    size_t out_row = (y - 1) * img_width;

    for (int x = 1; x < expanded_width - 1; x++) {
      int sum =
          temp[(y - 1) * expanded_width + x] + 2 * temp[y * expanded_width + x] + temp[(y + 1) * expanded_width + x];
      output[out_row + (x - 1)] = static_cast<uint8_t>(sum / 4);
    }
  });

  return true;
}

std::uint8_t KolotukhinAGaussinBlureTBB::GetPixel(const std::vector<std::uint8_t> &pixel_data, int img_width,
                                                  int img_height, int pos_x, int pos_y) {
  std::size_t x = static_cast<std::size_t>(std::max(0, std::min(pos_x, img_width - 1)));
  std::size_t y = static_cast<std::size_t>(std::max(0, std::min(pos_y, img_height - 1)));
  return pixel_data[(y * static_cast<std::size_t>(img_width)) + x];
}

bool KolotukhinAGaussinBlureTBB::PostProcessingImpl() {
  return true;
}

}  // namespace kolotukhin_a_gaussian_blur
