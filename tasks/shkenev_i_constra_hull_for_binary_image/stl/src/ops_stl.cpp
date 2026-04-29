#include "shkenev_i_constra_hull_for_binary_image/stl/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "util/include/util.hpp"

namespace shkenev_i_constra_hull_for_binary_image {

namespace {

constexpr uint8_t kThreshold = 128;

constexpr std::pair<int, int> dirs[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

inline bool InBounds(int x, int y, int w, int h) {
  return x >= 0 && x < w && y >= 0 && y < h;
}

inline int64_t Cross(const Point &a, const Point &b, const Point &c) {
  return int64_t(b.x - a.x) * (c.y - a.y) - int64_t(b.y - a.y) * (c.x - a.x);
}

}  // namespace

ShkenevIConstrHullSTL::ShkenevIConstrHullSTL(const InType &in) : work_(in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool ShkenevIConstrHullSTL::ValidationImpl() {
  const auto &in = GetInput();
  return in.width > 0 && in.height > 0 && in.pixels.size() == size_t(in.width) * size_t(in.height);
}

bool ShkenevIConstrHullSTL::PreProcessingImpl() {
  work_ = GetInput();
  ThresholdImage();
  return true;
}

void ShkenevIConstrHullSTL::ThresholdImage() {
  for (auto &p : work_.pixels) {
    p = (p > kThreshold) ? 255 : 0;
  }
}

size_t ShkenevIConstrHullSTL::Index(int x, int y, int w) {
  return size_t(y) * size_t(w) + size_t(x);
}

void ShkenevIConstrHullSTL::ExploreComponent(int sx, int sy, int w, int h, std::vector<uint8_t> &visited,
                                             std::vector<Point> &comp) {
  std::vector<Point> stack;
  stack.reserve(256);

  stack.emplace_back(sx, sy);
  visited[Index(sx, sy, w)] = 1;

  while (!stack.empty()) {
    Point cur = stack.back();
    stack.pop_back();

    comp.push_back(cur);

    for (auto [dx, dy] : dirs) {
      int nx = cur.x + dx;
      int ny = cur.y + dy;

      if (!InBounds(nx, ny, w, h)) {
        continue;
      }

      size_t idx = Index(nx, ny, w);

      if (visited[idx] || work_.pixels[idx] == 0) {
        continue;
      }

      visited[idx] = 1;
      stack.emplace_back(nx, ny);
    }
  }
}

void ShkenevIConstrHullSTL::FindComponents() {
  int w = work_.width;
  int h = work_.height;

  std::vector<uint8_t> visited(size_t(w) * size_t(h), 0);
  work_.components.clear();

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      size_t idx = Index(x, y, w);

      if (visited[idx] || work_.pixels[idx] == 0) {
        continue;
      }

      std::vector<Point> comp;
      ExploreComponent(x, y, w, h, visited, comp);

      if (!comp.empty()) {
        work_.components.emplace_back(std::move(comp));
      }
    }
  }
}

bool ShkenevIConstrHullSTL::RunImpl() {
  FindComponents();

  auto &comps = work_.components;
  auto &hulls = work_.convex_hulls;

  if (comps.empty()) {
    GetOutput() = work_;
    return true;
  }

  hulls.resize(comps.size());

  int num_threads = std::min<int>(ppc::util::GetNumThreads(), comps.size());
  std::vector<std::thread> threads;
  std::atomic<size_t> index{0};

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&]() {
      while (true) {
        size_t i = index.fetch_add(1, std::memory_order_relaxed);
        if (i >= comps.size()) {
          break;
        }

        const auto &comp = comps[i];

        if (comp.size() <= 2) {
          hulls[i] = comp;
        } else {
          hulls[i] = BuildHull(comp);
        }
      }
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  GetOutput() = work_;
  return true;
}

std::vector<Point> ShkenevIConstrHullSTL::BuildHull(const std::vector<Point> &pts_in) {
  std::vector<Point> pts = pts_in;

  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

  if (pts.size() <= 2) {
    return pts;
  }

  std::vector<Point> lower, upper;
  lower.reserve(pts.size());
  upper.reserve(pts.size());

  for (const auto &p : pts) {
    while (lower.size() >= 2 && Cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
      lower.pop_back();
    }
    lower.push_back(p);
  }

  for (auto it = pts.rbegin(); it != pts.rend(); ++it) {
    while (upper.size() >= 2 && Cross(upper[upper.size() - 2], upper.back(), *it) <= 0) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }

  lower.pop_back();
  upper.pop_back();

  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

bool ShkenevIConstrHullSTL::PostProcessingImpl() {
  return true;
}

}  // namespace shkenev_i_constra_hull_for_binary_image
