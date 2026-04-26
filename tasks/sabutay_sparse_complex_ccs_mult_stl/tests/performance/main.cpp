#include <gtest/gtest.h>

#include "sabutay_sparse_complex_ccs_mult_stl/all/include/ops_all.hpp"
#include "sabutay_sparse_complex_ccs_mult_stl/common/include/common.hpp"
#include "sabutay_sparse_complex_ccs_mult_stl/omp/include/ops_omp.hpp"
#include "sabutay_sparse_complex_ccs_mult_stl/seq/include/ops_seq.hpp"
#include "sabutay_sparse_complex_ccs_mult_stl/stl/include/ops_stl.hpp"
#include "sabutay_sparse_complex_ccs_mult_stl/tbb/include/ops_tbb.hpp"
#include "util/include/perf_test_util.hpp"

namespace sabutay_sparse_complex_ccs_mult_stl {

class SabutayRunPerfTestThreadsSTL : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 200;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SabutayRunPerfTestThreadsSTL, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NesterovATestTaskALL, NesterovATestTaskOMP, NesterovATestTaskSEQ,
                                SabutaySparseComplexCcsMultSTL, NesterovATestTaskTBB>(
        PPC_SETTINGS_sabutay_sparse_complex_ccs_mult_stl);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SabutayRunPerfTestThreadsSTL::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SabutayRunPerfTestThreadsSTL, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sabutay_sparse_complex_ccs_mult_stl
