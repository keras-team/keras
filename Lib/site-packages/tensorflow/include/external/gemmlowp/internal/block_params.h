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

// block_params.h: Logic to choose L1 and L2 block sizes
// to optimize cache-friendliness.

#ifndef GEMMLOWP_INTERNAL_BLOCK_PARAMS_H_
#define GEMMLOWP_INTERNAL_BLOCK_PARAMS_H_

#include "common.h"

namespace gemmlowp {

// A BlockParams instance contains a full description of all the block size
// parameters to be used by a Gemm.
// There are two nested levels of block subdivisions: first a subdivision
// into large blocks that should fit in last-level cache (what we call L2 here)
// and then another subdivision into smaller blocks that should fit in
// L1 cache. There is then actually a third level of subdivision to fit
// in registers, but we are not concerned with that here.
struct BlockParams {
  // L1 block parameters determine the size of small blocks that should
  // fit in L1 cache.
  int l1_rows;
  int l1_cols;
  int l1_depth;

  // L2 block parameters determine the size of larger blocks that should
  // fit in L2 cache.
  int l2_rows;
  int l2_cols;
  int l2_depth;

  template <typename KernelFormat>
  void Init(int rows, int cols, int depth, int num_threads, int l1_bytes_to_use,
            int l2_bytes_to_use, float l2_rhs_factor) {
    FindL2BlockSizes<KernelFormat>(rows, cols, depth, num_threads,
                                   l2_bytes_to_use, l2_rhs_factor, &l2_rows,
                                   &l2_cols, &l2_depth);
    FindL1BlockSizes<KernelFormat>(l2_rows, l2_cols, l2_depth, l1_bytes_to_use,
                                   &l1_rows, &l1_cols, &l1_depth);
  }

  template <typename KernelFormat>
  static void FindL2BlockSizes(int rows, int cols, int depth, int num_threads,
                               int l2_bytes_to_use, float l2_rhs_factor,
                               int* out_l2_rows, int* out_l2_cols,
                               int* out_l2_depth) {
    int l2_rows = 0;
    int l2_cols = 0;
    int l2_depth = 0;

    int per_thread_rows =
        std::max(1, RoundUp<KernelFormat::kRows>(rows) / num_threads);

    // No L2 blocking in the depth dimension at the moment.
    // Too much loss of accuracy due to storing intermediate results in
    // low precision.
    // However, we still want to round l2_depth up to the next multiple
    // of register size, so as to avoid having to special-case unaligned depths.
    l2_depth = RoundUp<kRegisterSize>(depth);

    {
      int max_cache_friendly_l2_cols = std::max(
          1, static_cast<int>(l2_rhs_factor * (l2_bytes_to_use / l2_depth)));
      int min_l2_cols_blocks =
          std::max(1, CeilQuotient(cols, max_cache_friendly_l2_cols));
      l2_cols =
          RoundUp<KernelFormat::kCols>(CeilQuotient(cols, min_l2_cols_blocks));
    }

    // No L2 blocking in the row dimension if l2_rhs_factor is 1.0 as the row
    // dimension concerns only the LHS. Blocking only RHS matrix for L2 enhances
    // the performance on x86.
    if (l2_rhs_factor == 1.0f) {
      l2_rows = RoundUp<KernelFormat::kRows>(per_thread_rows);
    } else {
      int max_cache_friendly_l2_rows =
          std::max(1, (l2_bytes_to_use - l2_depth * l2_cols) /
                          (num_threads * (l2_depth + 4 * l2_cols)));
      int min_l2_rows_blocks = std::max(
          1, CeilQuotient(per_thread_rows, max_cache_friendly_l2_rows));
      l2_rows = RoundUp<KernelFormat::kRows>(
          CeilQuotient(per_thread_rows, min_l2_rows_blocks));
    }

    *out_l2_rows = l2_rows;
    *out_l2_cols = l2_cols;
    *out_l2_depth = l2_depth;
  }

  template <typename KernelFormat>
  static void FindL1BlockSizes(int rows, int cols, int depth,
                               int l1_bytes_to_use, int* out_l1_rows,
                               int* out_l1_cols, int* out_l1_depth) {
    int l1_rows = 0;
    int l1_cols = 0;
    int l1_depth = 0;

    // L2 block sizes should already be multiples of kernel block sizes.
    assert(rows % KernelFormat::kRows == 0);
    assert(cols % KernelFormat::kCols == 0);
    assert(depth % KernelFormat::kDepth == 0);

    // No L1 blocking in the columns dimension at the moment.
    // Thought not to be needed. Similar to Eigen.
    l1_cols = cols;

    {
      int max_cache_friendly_l1_depth = std::max(
          1, (l1_bytes_to_use - 4 * KernelFormat::kRows * KernelFormat::kCols) /
                 (KernelFormat::kRows + KernelFormat::kCols));
      int min_l1_depth_blocks =
          std::max(1, CeilQuotient(depth, max_cache_friendly_l1_depth));
      l1_depth =
          RoundUp<kRegisterSize>(CeilQuotient(depth, min_l1_depth_blocks));
    }

    {
      int max_cache_friendly_l1_rows =
          std::max(1, l1_bytes_to_use / (l1_depth + 4 * l1_cols));
      int min_l1_rows_blocks =
          std::max(1, CeilQuotient(rows, max_cache_friendly_l1_rows));
      l1_rows =
          RoundUp<KernelFormat::kRows>(CeilQuotient(rows, min_l1_rows_blocks));
    }

    *out_l1_rows = l1_rows;
    *out_l1_cols = l1_cols;
    *out_l1_depth = l1_depth;
  }
};

// A SideBlockParams instance contains only the block params relevant to
// one side (LHS or RHS), expressed in terms of 'width' instead of
// rows/colums. See the explanation in kernel.h: in the LHS, 'width' means
// the number of rows, while in the RHS, 'width' means the number of columns.
// That allows us to write generic code that applies to either LHS or RHS.
struct SideBlockParams {
  // L1 block parameters determine the size of small blocks that should
  // fit in L1 cache.
  int l1_width;
  int l1_depth;

  // L2 block parameters determine the size of larger blocks that should
  // fit in L2 cache.
  int l2_width;
  int l2_depth;
};

enum class Side { Lhs, Rhs };

inline void GetSideBlockParams(Side side, SideBlockParams* side_block_params,
                               const BlockParams& block_params) {
  side_block_params->l1_width =
      side == Side::Lhs ? block_params.l1_rows : block_params.l1_cols;
  side_block_params->l2_width =
      side == Side::Lhs ? block_params.l2_rows : block_params.l2_cols;

  side_block_params->l1_depth = block_params.l1_depth;
  side_block_params->l2_depth = block_params.l2_depth;
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_BLOCK_PARAMS_H_
