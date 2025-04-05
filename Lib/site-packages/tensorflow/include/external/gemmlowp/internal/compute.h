// Copyright 2015 The Gemmlowp Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// compute.h: the central stage of the Gemm computation, operates
// on already-packed LHS and RHS blocks and calls the Gemm kernel
// to compute a block of the product.

#ifndef GEMMLOWP_INTERNAL_COMPUTE_H_
#define GEMMLOWP_INTERNAL_COMPUTE_H_

#include "block_params.h"
#include "kernel.h"
#include "pack.h"

namespace gemmlowp {

template <typename PackedLhs, typename PackedRhs, typename PackedResult>
class ComputeImpl {
  typedef typename PackedLhs::KernelSideFormat KernelLhsFormat;
  typedef typename PackedRhs::KernelSideFormat KernelRhsFormat;
  typedef KernelFormat<KernelLhsFormat, KernelRhsFormat> Format;

  const KernelBase& kernel_;
  const BlockParams& block_params_;

  PackedResult* const packed_result_;
  const PackedLhs& packed_lhs_;
  const PackedRhs& packed_rhs_;

 public:
  ComputeImpl(const KernelBase& _kernel, const BlockParams& _block_params,
              PackedResult* _packed_result, const PackedLhs& _packed_lhs,
              const PackedRhs& _packed_rhs)
      : kernel_(_kernel),
        block_params_(_block_params),
        packed_result_(_packed_result),
        packed_lhs_(_packed_lhs),
        packed_rhs_(_packed_rhs) {}

  void Compute(int depth) {
    depth = RoundUp<Format::kDepth>(depth);
    assert(depth <= block_params_.l2_depth);
    for (int d = 0; d < depth; d += block_params_.l1_depth) {
      int ds = std::min(block_params_.l1_depth, depth - d);

      for (int r = 0; r < block_params_.l2_rows; r += block_params_.l1_rows) {
        int rs = std::min(block_params_.l1_rows, block_params_.l2_rows - r);

        ComputeL1(r, rs, 0, block_params_.l2_cols, d, ds);
      }
    }
  }

 private:
  static void MarkPackedResultBlockAsInitialized(
      const MatrixMap<std::int32_t, MapOrder::ColMajor>& packed_result_block) {
#ifdef GEMMLOWP_MARK_MEMORY_AS_INITIALIZED
    for (int col = 0; col < packed_result_block.cols(); col++) {
      MarkMemoryAsInitialized(
          packed_result_block.data() + col * packed_result_block.cols_stride(),
          packed_result_block.rows());
    }
#else
    (void)packed_result_block;
#endif
  }

  void ComputeRun(int start_row, int start_col, int start_depth,
                  int depth) GEMMLOWP_NOINLINE {
    packed_lhs_.seek_run(start_row, start_depth);
    packed_rhs_.seek_run(start_col, start_depth);
    auto packed_result_block = packed_result_->Map().block(
        start_row, start_col, Format::kRows, Format::kCols);
    kernel_.Run(packed_result_block.data(), packed_result_block.rows_stride(),
                packed_result_block.cols_stride(), packed_lhs_.current_data(),
                packed_rhs_.current_data(), start_depth, depth);
    MarkPackedResultBlockAsInitialized(packed_result_block);
  }

  void ComputeL1(int start_row, int rows, int start_col, int cols,
                 int start_depth, int depth) {
    assert(rows % Format::kRows == 0);
    assert(cols % Format::kCols == 0);
    assert(depth % Format::kDepth == 0);

    for (int c = 0; c < cols; c += Format::kCols) {
      for (int r = 0; r < rows; r += Format::kRows) {
        ComputeRun(start_row + r, start_col + c, start_depth, depth);
      }
    }
  }
};

template <typename PackedLhs, typename PackedRhs, typename PackedResult>
void Compute(const KernelBase& kernel, const BlockParams& block_params,
             PackedResult* packed_result, const PackedLhs& packed_lhs,
             const PackedRhs& packed_rhs, int depth) {
  ScopedProfilingLabel label("compute");
  ComputeImpl<PackedLhs, PackedRhs, PackedResult> impl(
      kernel, block_params, packed_result, packed_lhs, packed_rhs);

  impl.Compute(depth);
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_COMPUTE_H_
