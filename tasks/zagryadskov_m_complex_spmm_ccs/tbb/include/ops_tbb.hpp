#pragma once

#include <tbb/enumerable_thread_specific.h>

#include <complex>
#include <vector>

#include "task/include/task.hpp"
#include "zagryadskov_m_complex_spmm_ccs/common/include/common.hpp"

namespace zagryadskov_m_complex_spmm_ccs {

class ZagryadskovMComplexSpMMCCSTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }
  explicit ZagryadskovMComplexSpMMCCSTBB(const InType &in);

 private:
  static void SpMM(const CCS &a, const CCS &b, CCS &c);
  static void SpMM_symbolic(const CCS &a, const CCS &b, std::vector<int> &col_ptr);
  static void SpMM_numeric(const CCS &a, const CCS &b, CCS &c, const std::complex<double> &zero, double eps);
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace zagryadskov_m_complex_spmm_ccs
