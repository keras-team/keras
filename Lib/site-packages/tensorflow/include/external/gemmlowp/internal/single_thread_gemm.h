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

// single_thread_gemm.h: Single-threaded GEMM implementation.
// This is a good place to start reading code, as it shows the overall
// structure of a GEMM and is much simpler than multi_thread_gemm.h.

#ifndef GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_
#define GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_

#include <cassert>

#include "../public/map.h"
#include "allocator.h"
#include "compute.h"
#include "kernel.h"
#include "pack.h"
#include "unpack.h"

#ifdef GEMMLOWP_PROFILING_SIZES
#ifndef GEMMLOWP_PROFILING
#error GEMMLOWP_PROFILING_SIZES without GEMMLOWP_PROFILING
#endif
#include <string>
#include <unordered_map>
#endif

namespace gemmlowp {

class SingleThreadGemmContext {
 public:
  Allocator* allocator() { return &allocator_; }

  void set_l1_bytes_to_use(int n) { l1_bytes_to_use_ = n; }
  void set_l2_bytes_to_use(int n) { l2_bytes_to_use_ = n; }
  void set_l2_rhs_factor(float n) { l2_rhs_factor_ = n; }

  int l1_bytes_to_use() const { return l1_bytes_to_use_; }
  int l2_bytes_to_use() const { return l2_bytes_to_use_; }
  float l2_rhs_factor() const { return l2_rhs_factor_; }

 protected:
  Allocator allocator_;

  // The cache configurationt to use.
  int l1_bytes_to_use_ = kDefaultL1CacheSize;
  int l2_bytes_to_use_ = kDefaultL2CacheSize;
  float l2_rhs_factor_ = kDefaultL2RhsFactor;
};

template <typename KernelFormat, typename InputScalar, typename OutputScalar,
          typename BitDepthParams, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
void SingleThreadGemm(SingleThreadGemmContext* context,
                      const KernelBase& kernel,
                      const MatrixMap<const InputScalar, LhsOrder>& lhs,
                      const MatrixMap<const InputScalar, RhsOrder>& rhs,
                      MatrixMap<OutputScalar, ResultOrder>* result,
                      const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                      const OutputPipelineType& output_pipeline) {
  ScopedProfilingLabel label("gemmlowp::SingleThreadGemm");

  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  // zero sizes should have been caught earlier and early-returned.
  assert(rows > 0);
  assert(cols > 0);
  assert(depth > 0);

  // The case of rows<cols should have been caught earlier and transposed.
  assert(rows >= cols);

  Allocator* allocator = context->allocator();

  BlockParams block_params;
  block_params.Init<KernelFormat>(
      rows, cols, depth, 1, context->l1_bytes_to_use(),
      context->l2_bytes_to_use(), context->l2_rhs_factor());

#ifdef GEMMLOWP_PROFILING_SIZES
  // Using a static map of label strings. Not reentrant at all!
  static std::unordered_map<std::uint64_t, std::string> labels_map;
  std::uint64_t sizes_hash = static_cast<std::uint64_t>(rows) ^
                             (static_cast<std::uint64_t>(depth) << 16) ^
                             (static_cast<std::uint64_t>(cols) << 32);
  if (!labels_map.count(sizes_hash)) {
    char label[256];
    snprintf(label, sizeof(label),
             "(rows = %d, depth = %d, cols = %d, l2_rows = %d, l2_depth = %d, "
             "l2_cols = %d, l1_rows = %d, l1_depth = %d, l1_cols = %d)",
             rows, depth, cols, block_params.l2_rows, block_params.l2_depth,
             block_params.l2_cols, block_params.l1_rows, block_params.l1_depth,
             block_params.l1_cols);
    labels_map[sizes_hash] = label;
  }
  ScopedProfilingLabel size_label(labels_map[sizes_hash].c_str());
#endif

  PackedSideBlock<typename KernelFormat::Lhs> packed_lhs(Side::Lhs, allocator,
                                                         block_params);
  PackedSideBlock<typename KernelFormat::Rhs> packed_rhs(Side::Rhs, allocator,
                                                         block_params);

  PackedResult packed_result(allocator, block_params);

  allocator->Commit();

  const bool pack_rhs_once = block_params.l2_cols >= cols;

  if (pack_rhs_once) {
    PackRhs(&packed_rhs, rhs);
  }

  for (int r = 0; r < rows; r += block_params.l2_rows) {
    int rs = std::min(block_params.l2_rows, rows - r);

    PackLhs(&packed_lhs, lhs.block(r, 0, rs, depth));

    for (int c = 0; c < cols; c += block_params.l2_cols) {
      int cs = std::min(block_params.l2_cols, cols - c);

      if (!pack_rhs_once) {
        PackRhs(&packed_rhs, rhs.block(0, c, depth, cs));
      }

      Compute(kernel, block_params, &packed_result, packed_lhs, packed_rhs,
              depth);

      UnpackResult<KernelFormat>(
          result, MatrixBlockBounds(r, c, rs, cs), packed_result, depth,
          packed_lhs.sums_of_each_slice(), packed_rhs.sums_of_each_slice(),
          lhs_offset.block(r, rs), rhs_offset.block(c, cs), output_pipeline);
    }
  }

  allocator->Decommit();
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_
