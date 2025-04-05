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

// unpack.h: unpacking the result blocks computed by compute.h,
// storing them into the destination matrix.

#ifndef GEMMLOWP_INTERNAL_UNPACK_H_
#define GEMMLOWP_INTERNAL_UNPACK_H_

#include "allocator.h"
#include "block_params.h"
#include "output.h"
#include "pack.h"

#include <cmath>

namespace gemmlowp {

class PackedResult {
 public:
  PackedResult(Allocator* _allocator, const BlockParams& _block_params)
      : allocator_(_allocator), block_params_(_block_params) {
    matrix_handle_ = allocator_->Reserve<std::int32_t>(block_params_.l2_rows *
                                                       block_params_.l2_cols);
  }

  ~PackedResult() {}

  MatrixMap<std::int32_t, MapOrder::ColMajor> Map() {
    return MatrixMap<std::int32_t, MapOrder::ColMajor>(
        allocator_->GetPointer<std::int32_t>(matrix_handle_),
        block_params_.l2_rows, block_params_.l2_cols, block_params_.l2_rows);
  }

  MatrixMap<const std::int32_t, MapOrder::ColMajor> Map() const {
    return MatrixMap<const std::int32_t, MapOrder::ColMajor>(
        allocator_->GetPointer<const std::int32_t>(matrix_handle_),
        block_params_.l2_rows, block_params_.l2_cols, block_params_.l2_rows);
  }

 private:
  Allocator* allocator_;
  Allocator::Handle matrix_handle_;
  const BlockParams& block_params_;
};

struct MatrixBlockBounds {
  int start_row;
  int start_col;
  int rows;
  int cols;

  MatrixBlockBounds(int start_row_, int start_col_, int rows_, int cols_)
      : start_row(start_row_),
        start_col(start_col_),
        rows(rows_),
        cols(cols_) {}
};

template <int Rows, int Cols, typename SrcMapType>
void PrefetchResultBlock(const SrcMapType& src,
                         const VectorMap<const std::int32_t, VectorShape::Col>&
                             lhs_sums_of_each_slice,
                         int src_row, int src_col) {
  const std::int32_t* src_data = src.data(src_row, src_col);
  const int src_stride = src.stride();
  const std::int32_t* lhs_sums_data = lhs_sums_of_each_slice.data(src_row);
  for (int r = 0; r < Rows; r += 4) {
    Prefetch(lhs_sums_data + r);
  }
  for (int c = 0; c < Cols; c++) {
    for (int r = 0; r < Rows; r += 4) {
      Prefetch(src_data + r + c * src_stride);
    }
  }
}

template <typename KernelFormat, typename RegisterBlockType,
          typename SrcMapType, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineExecutorType, typename DstType>
void UnpackResultBlock(const SrcMapType& src,
                       const OutputPipelineExecutorType& executor, DstType* dst,
                       const VectorMap<const std::int32_t, VectorShape::Col>&
                           lhs_sums_of_each_slice,
                       const VectorMap<const std::int32_t, VectorShape::Row>&
                           rhs_sums_of_each_slice,
                       const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                       int depth, int src_row, int src_col, int src_global_row,
                       int src_global_col, int dst_row, int dst_col) {
  using KernelLhsInputScalar = typename KernelFormat::Lhs::InputScalar;
  using KernelLhsScalar = typename KernelFormat::Lhs::Scalar;
  using KernelRhsInputScalar = typename KernelFormat::Rhs::InputScalar;
  using KernelRhsScalar = typename KernelFormat::Rhs::Scalar;
  static constexpr int KernelLhsZeroPointInput =
      ZeroPointInputValue<KernelLhsInputScalar, KernelLhsScalar>::kValue;
  static constexpr int KernelRhsZeroPointInput =
      ZeroPointInputValue<KernelRhsInputScalar, KernelRhsScalar>::kValue;
  auto acc = Load<RegisterBlockType>(src, src_row, src_col);
  const auto& lhs_sums_of_each_slice_block =
      LoadForBroadcasting<RegisterBlockType>(lhs_sums_of_each_slice, src_row);
  const auto& rhs_sums_of_each_slice_block =
      LoadForBroadcasting<RegisterBlockType>(rhs_sums_of_each_slice, src_col);
  auto lhs_offset_block =
      LoadForBroadcasting<RegisterBlockType>(lhs_offset, src_row);
  auto rhs_offset_block =
      LoadForBroadcasting<RegisterBlockType>(rhs_offset, src_col);
  AddConstant<KernelLhsZeroPointInput>(&lhs_offset_block);
  AddConstant<KernelRhsZeroPointInput>(&rhs_offset_block);
  BroadcastMulAdd(lhs_sums_of_each_slice_block, rhs_offset_block, &acc);
  for (int i = 0; i < decltype(rhs_offset_block)::kRegisterCount; i++) {
    rhs_offset_block.buf.reg[i] = Mul(rhs_offset_block.buf.reg[i], depth);
  }
  BroadcastMulAdd(BroadcastAdd(rhs_sums_of_each_slice_block, rhs_offset_block),
                  lhs_offset_block, &acc);
  executor.Execute(acc, dst, src_global_row, src_global_col, dst_row, dst_col);
}

template <typename KernelFormat, typename ResultBlockType,
          typename PackedResultType, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
void UnpackResult(ResultBlockType* dst, const MatrixBlockBounds& dst_block,
                  const PackedResultType& src, int depth,
                  const std::int32_t* lhs_sums_of_each_slice_ptr,
                  const std::int32_t* rhs_sums_of_each_slice_ptr,
                  const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                  const OutputPipelineType& output_pipeline) {
  ScopedProfilingLabel label(ResultBlockType::kOrder == MapOrder::ColMajor
                                 ? "unpack to column-major"
                                 : "unpack to row-major");
  assert(dst_block.start_row >= 0);
  assert(dst_block.start_row + dst_block.rows <= dst->rows());
  assert(dst_block.start_col >= 0);
  assert(dst_block.start_col + dst_block.cols <= dst->cols());
  const auto src_map = src.Map();
  const VectorMap<const std::int32_t, VectorShape::Col> lhs_sums_of_each_slice(
      lhs_sums_of_each_slice_ptr, dst_block.rows);
  const VectorMap<const std::int32_t, VectorShape::Row> rhs_sums_of_each_slice(
      rhs_sums_of_each_slice_ptr, dst_block.cols);
  using Int32x1x1 = RegisterBlock<std::int32_t, 1, 1>;
  using Int32x4x1 = RegisterBlock<std::int32_t, 4, 1>;
  using Int32x8x1 = RegisterBlock<std::int32_t, 8, 1>;
  using Int32x1x4 = RegisterBlock<std::int32_t, 1, 4>;
  using Int32x4x4 = RegisterBlock<std::int32_t, 4, 4>;
  using Int32x8x4 = RegisterBlock<std::int32_t, 8, 4>;

  using DstScalarType = typename ResultBlockType::Scalar;
  using DstScalarx8x8 = RegisterBlock<DstScalarType, 8, 8>;

  OutputPipelineExecutor<OutputPipelineType, Int32x1x1>
      output_pipeline_executor_1x1(output_pipeline);
  OutputPipelineExecutor<OutputPipelineType, Int32x4x1>
      output_pipeline_executor_4x1(output_pipeline);
  OutputPipelineExecutor<OutputPipelineType, Int32x8x1>
      output_pipeline_executor_8x1(output_pipeline);
  OutputPipelineExecutor<OutputPipelineType, Int32x1x4>
      output_pipeline_executor_1x4(output_pipeline);
  OutputPipelineExecutor<OutputPipelineType, Int32x4x4>
      output_pipeline_executor_4x4(output_pipeline);
  OutputPipelineExecutor<OutputPipelineType, Int32x8x4>
      output_pipeline_executor_8x4(output_pipeline);

  int c8 = 0;
  if (ResultBlockType::kOrder == MapOrder::RowMajor) {
    for (; c8 <= dst_block.cols - 8; c8 += 8) {
      PrefetchResultBlock<8, 8>(src_map, lhs_sums_of_each_slice, 0, c8);
      int r = 0;
      for (; r <= dst_block.rows - 8; r += 8) {
        const int global_row = r + dst_block.start_row;
        PrefetchResultBlock<8, 8>(src_map, lhs_sums_of_each_slice, r + 8, c8);
        DstScalarType dst_colmajor_buf[64];
        MatrixMap<DstScalarType, MapOrder::ColMajor> dst_colmajor_map(
            dst_colmajor_buf, 8, 8);
        for (int cx = 0; cx < 8; cx += 4) {
          const int c = c8 + cx;
          const int global_col = c + dst_block.start_col;
          UnpackResultBlock<KernelFormat, Int32x8x4>(
              src_map, output_pipeline_executor_8x4, &dst_colmajor_map,
              lhs_sums_of_each_slice, rhs_sums_of_each_slice, lhs_offset,
              rhs_offset, depth, r, c, global_row, global_col, 0, cx);
        }
        StoreFinalOutput(LoadContiguous<DstScalarx8x8>(dst_colmajor_buf), dst,
                         r + dst_block.start_row, c8 + dst_block.start_col);
      }
      for (; r <= dst_block.rows - 4; r += 4) {
        const int global_row = r + dst_block.start_row;
        for (int cx = 0; cx < 8; cx += 4) {
          const int c = c8 + cx;
          const int global_col = c + dst_block.start_col;
          UnpackResultBlock<KernelFormat, Int32x4x4>(
              src_map, output_pipeline_executor_4x4, dst,
              lhs_sums_of_each_slice, rhs_sums_of_each_slice, lhs_offset,
              rhs_offset, depth, r, c, global_row, global_col, global_row,
              global_col);
        }
      }
      for (; r < dst_block.rows; r++) {
        const int global_row = r + dst_block.start_row;
        for (int cx = 0; cx < 8; cx += 4) {
          const int c = c8 + cx;
          const int global_col = c + dst_block.start_col;
          UnpackResultBlock<KernelFormat, Int32x1x4>(
              src_map, output_pipeline_executor_1x4, dst,
              lhs_sums_of_each_slice, rhs_sums_of_each_slice, lhs_offset,
              rhs_offset, depth, r, c, global_row, global_col, global_row,
              global_col);
        }
      }
    }
  }
  int c = c8;
  for (; c <= dst_block.cols - 4; c += 4) {
    const int global_col = c + dst_block.start_col;
    PrefetchResultBlock<8, 4>(src_map, lhs_sums_of_each_slice, 0, c);
    int r = 0;
    for (; r <= dst_block.rows - 8; r += 8) {
      const int global_row = r + dst_block.start_row;
      PrefetchResultBlock<8, 4>(src_map, lhs_sums_of_each_slice, r + 8, c);
      UnpackResultBlock<KernelFormat, Int32x8x4>(
          src_map, output_pipeline_executor_8x4, dst, lhs_sums_of_each_slice,
          rhs_sums_of_each_slice, lhs_offset, rhs_offset, depth, r, c,
          global_row, global_col, global_row, global_col);
    }
    for (; r <= dst_block.rows - 4; r += 4) {
      const int global_row = r + dst_block.start_row;
      UnpackResultBlock<KernelFormat, Int32x4x4>(
          src_map, output_pipeline_executor_4x4, dst, lhs_sums_of_each_slice,
          rhs_sums_of_each_slice, lhs_offset, rhs_offset, depth, r, c,
          global_row, global_col, global_row, global_col);
    }
    for (; r < dst_block.rows; r++) {
      const int global_row = r + dst_block.start_row;
      UnpackResultBlock<KernelFormat, Int32x1x4>(
          src_map, output_pipeline_executor_1x4, dst, lhs_sums_of_each_slice,
          rhs_sums_of_each_slice, lhs_offset, rhs_offset, depth, r, c,
          global_row, global_col, global_row, global_col);
    }
  }
  for (; c < dst_block.cols; c++) {
    const int global_col = c + dst_block.start_col;
    PrefetchResultBlock<8, 1>(src_map, lhs_sums_of_each_slice, 0, c);
    int r = 0;
    for (; r <= dst_block.rows - 8; r += 8) {
      const int global_row = r + dst_block.start_row;
      PrefetchResultBlock<8, 1>(src_map, lhs_sums_of_each_slice, r + 8, c);
      UnpackResultBlock<KernelFormat, Int32x8x1>(
          src_map, output_pipeline_executor_8x1, dst, lhs_sums_of_each_slice,
          rhs_sums_of_each_slice, lhs_offset, rhs_offset, depth, r, c,
          global_row, global_col, global_row, global_col);
    }
    for (; r <= dst_block.rows - 4; r += 4) {
      const int global_row = r + dst_block.start_row;
      UnpackResultBlock<KernelFormat, Int32x4x1>(
          src_map, output_pipeline_executor_4x1, dst, lhs_sums_of_each_slice,
          rhs_sums_of_each_slice, lhs_offset, rhs_offset, depth, r, c,
          global_row, global_col, global_row, global_col);
    }
    for (; r < dst_block.rows; r++) {
      const int global_row = r + dst_block.start_row;
      UnpackResultBlock<KernelFormat, Int32x1x1>(
          src_map, output_pipeline_executor_1x1, dst, lhs_sums_of_each_slice,
          rhs_sums_of_each_slice, lhs_offset, rhs_offset, depth, r, c,
          global_row, global_col, global_row, global_col);
    }
  }
}

}  // end namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_UNPACK_H_
