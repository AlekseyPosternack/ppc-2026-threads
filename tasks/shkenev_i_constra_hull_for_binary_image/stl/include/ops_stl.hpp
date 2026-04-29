#pragma once

#include <thread>
#include <vector>

#include "shkenev_i_constra_hull_for_binary_image/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shkenev_i_constra_hull_for_binary_image {

class ShkenevIConstrHullSTL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSTL;
  }

  explicit ShkenevIConstrHullSTL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void ThresholdImage();
  void FindComponents();
  void ExploreComponent(int sx, int sy, int w, int h, std::vector<uint8_t> &visited, std::vector<Point> &comp);

  static std::vector<Point> BuildHull(const std::vector<Point> &points);
  static size_t Index(int x, int y, int width);

  BinaryImage work_;
};

}  // namespace shkenev_i_constra_hull_for_binary_image
