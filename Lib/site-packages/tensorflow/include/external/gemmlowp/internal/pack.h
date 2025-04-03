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

// pack.h: packing blocks of the LHS and RHS into the data layout
// that is expected by compute.h and eventually by kernels.
// Because this data layout depends on the kernel format, code here
// is templated in KernelLhsFormat/KernelRhsFormat.
//
// Readers note: an important theme around here is that we try hard
// to handle both Lhs and Rhs with a single piece of code. We indifferently
// refer to the Lhs and Rhs as a 'Side'. Instead of addressing matrices
// by (row, column) indices, we address them by (width, depth), as explained
// in kernel.h. This allows us to handle both Lhs and Rhs on an equal footing,
// at once.

#ifndef GEMMLOWP_INTERNAL_PACK_H_
#define GEMMLOWP_INTERNAL_PACK_H_

#include <cstring>

#include "allocator.h"
#include "block_params.h"
#include "common.h"
#include "kernel.h"

namespace gemmlowp {

// A PackedSideBlock instance is a packed block of either the LHS or RHS
// (whence the generic 'Side' name).
//
// 'Packed' means that it is laid out in the storage order that
// is expected by the specified kernel format. From a block of the input
// LHS or RHS matrix, one obtains a PackedSideBlock by calling PackLhs()
// or PackRhs().
template <typename tKernelSideFormat>
class PackedSideBlock {
 public:
  typedef tKernelSideFormat KernelSideFormat;

  PackedSideBlock(Side side, Allocator* allocator,
                  const BlockParams& block_params)
      : allocator_(allocator), pos_(0) {
    GetSideBlockParams(side, &params_, block_params);
    data_handle_ =
        allocator_->Reserve<std::uint8_t>(params_.l2_width * params_.l2_depth);
    sums_of_each_slice_handle_ =
        allocator_->Reserve<std::int32_t>(params_.l2_width);
  }

  ~PackedSideBlock() {}

  void seek_run(int start_width, int start_depth) const {
    int kernel_run_depth =
        std::min<int>(params_.l1_depth, params_.l2_depth - start_depth);
    pos_ = params_.l2_width * start_depth + start_width * kernel_run_depth;
  }

  void seek_next_cell() const { pos_ += KernelSideFormat::Cell::kSize; }

  void seek_forward_n_cells(int n) const {
    pos_ += n * KernelSideFormat::Cell::kSize;
  }

  // TODO(suharshs): The datatype can now be int8 as well. We could introduce a
  // new int8 current_data impl as well. This change would propagate to all pack
  // impls and the Kernel::Run API, which all assume uint8. For now we leave
  // this as-is pending future refactor.
  const std::uint8_t* current_data() const {
    return allocator_->GetPointer<std::uint8_t>(data_handle_) + pos_;
  }

  std::uint8_t* current_data() {
    return allocator_->GetPointer<std::uint8_t>(data_handle_) + pos_;
  }

  std::int32_t* sums_of_each_slice() {
    return allocator_->GetPointer<std::int32_t>(sums_of_each_slice_handle_);
  }

  const std::int32_t* sums_of_each_slice() const {
    return allocator_->GetPointer<const std::int32_t>(
        sums_of_each_slice_handle_);
  }

  const SideBlockParams& params() const { return params_; }

 private:
  // The block size parameters that this PackedSizeBlock follows.
  // The L2 parameters determine its overall size, while the L1 parameters,
  // together with the kernel format template parameter, determine
  // the fine details of the storage/traversal order.
  SideBlockParams params_;

  // Pointer to the allocator provided by the caller. Not owned.
  // The Allocator is assumed to outlive the PackedSideBlock.
  Allocator* const allocator_;

  // Handle on the buffer backing this packed block. Owned.
  Allocator::Handle data_handle_;

  // Handle on the additional buffer backing the vector of sums of slices
  // associated with this block. Owned.
  Allocator::Handle sums_of_each_slice_handle_;

  // pos_ is the current position in the buffer, which we access
  // sequentially, like a file.
  // The idea is that we pack data in the same order as it is
  // going to be traversed during the computation, which for
  // cache-friendliness reasons is complicated to random-access,
  // as the offsets calculations would be intricate. So we
  // give up random-access addressing, and instead content ourselves
  // with sequential access.
  //
  // pos_ is mutable because during the computation we will want to
  // be able to iterate on the data in a const PackedSideBlock.
  mutable int pos_;
};

// WidthMajor and DepthMajor are custom phrases modelled after the
// standard terminology 'row-major' and 'column-major'. Their meaning
// should be transparent once one has read the explanation in kernel.h:
// for example, in the Lhs, the 'width' dimension is the rows dimension,
// so there WidthMajor means RowMajor, while in the Rhs it is the opposite.
// Another way to put it: WidthMajor means that contiguous storage is used
// for entries having the same 'width' index.
enum class SideMapOrder { WidthMajor, DepthMajor };

// Similar to MatrixMap from map.h, but in terms of width/depth instead of
// rows/columns. Used to address blocks of the input LHS/RHS matrices when
// packing them.
template <typename tScalar, SideMapOrder tOrder>
class SideMap {
 public:
  typedef tScalar Scalar;
  static constexpr SideMapOrder kOrder = tOrder;

  SideMap(Scalar* data, int width, int depth, int stride)
      : data_(data), width_(width), depth_(depth), stride_(stride) {}

  SideMap(Scalar* data, int width, int depth)
      : data_(data), width_(width), depth_(depth) {
    stride_ = kOrder == SideMapOrder::WidthMajor ? depth_ : width_;
  }

  SideMap(const SideMap& other) = default;
  SideMap& operator=(const SideMap& other) = default;

  int width() const { return width_; }
  int depth() const { return depth_; }
  int stride() const { return stride_; }
  int width_stride() const {
    return kOrder == SideMapOrder::DepthMajor ? 1 : stride_;
  }
  int depth_stride() const {
    return kOrder == SideMapOrder::WidthMajor ? 1 : stride_;
  }
  Scalar* data() const { return data_; }
  Scalar* data(int w, int d) const {
    return data_ + w * width_stride() + d * depth_stride();
  }
  Scalar operator()(int w, int d) const { return *data(w, d); }
  Scalar& operator()(int w, int d) { return *data(w, d); }

  SideMap block(int start_width, int start_depth, int block_width,
                int block_depth) const {
    assert(start_width >= 0);
    assert(start_width + block_width <= width_);
    assert(start_depth >= 0);
    assert(start_depth + block_depth <= depth_);

    return SideMap(data(start_width, start_depth), block_width, block_depth,
                   stride_);
  }

 private:
  Scalar* data_;  // not owned.
  int width_, depth_, stride_;
};

// A PackingRegisterBlock is a small fixed-size block of a matrix being
// packed. This class is the generic non-optimized implementation,
// it is inherited by the generic implementation of PackingRegisterBlock,
// which may be overriden by template specialization. Overriding it is how
// one may provide optimized packing code paths.
//
// The packing of a block proceeds in two steps:
//   1. Ensuring that we have a complete block of source data, i.e. a block of
//      the compile-time prescribed size. This is where we handle unaligned
//      boundaries: if we don't have a complete block of source data, then
//      we copy and zero-extend it into a local temporary (complete_src_),
//      see MakeCompleteSrc. In the generic case, we do have a complete block,
//      so we just use it in-place, see UseCompleteSrcInPlace.
//   2. Packing a complete block into the destination, see Pack. This is the
//      most critical part, so it's convenient that unaligned boundaries have
//      already been handled in step 1.
template <typename SrcMapType, typename PackedSideBlock>
class PackingRegisterBlockBase {
 public:
  typedef typename PackedSideBlock::KernelSideFormat KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  typedef typename KernelSideFormat::InputScalar KernelInputScalar;
  typedef typename KernelSideFormat::Scalar KernelScalar;
  static constexpr int kCells = KernelSideFormat::kCells;
  static constexpr int kCellWidth = CellFormat::kWidth;
  static constexpr int kKernelWidth = CellFormat::kWidth * kCells;
  static constexpr int kCellDepth = CellFormat::kDepth;
  static constexpr int kCellSize = CellFormat::kSize;
  static constexpr SideMapOrder kSrcOrder = SrcMapType::kOrder;
  static constexpr int kZeroPointInputValue =
      ZeroPointInputValue<KernelInputScalar, KernelScalar>::kValue;

  PackingRegisterBlockBase() : complete_src_(nullptr, 0, 0, 0) {}

 protected:
  // The source data that's ready for packing. May point to
  // in-place actual source data if it's already a complete block,
  // (see UseCompleteSrcInPlace)
  // or to the local buf_ below into which we copy incomplete blocks
  // (see MakeCompleteSrc)
  SrcMapType complete_src_;

  // Temporary buffer for loading incomplete blocks to,
  // in the source storage order
  std::uint8_t buf_[kKernelWidth * kRegisterSize];

 public:
  // Selects a block if in-place source data that's already a complete block.
  void UseCompleteSrcInPlace(const SrcMapType& src) { complete_src_ = src; }
  // Copies an incomplete block of source data into a local temporary
  // complete block by zero-extending it.
  void MakeCompleteSrc(const SrcMapType& src) {
    memset(buf_, kZeroPointInputValue, kKernelWidth * kRegisterSize);
    if (kSrcOrder == SideMapOrder::WidthMajor) {
      for (int w = 0; w < src.width(); w++) {
        memcpy(buf_ + w * kRegisterSize, src.data(w, 0), src.depth());
      }
    } else {
      assert(kSrcOrder == SideMapOrder::DepthMajor);
      for (int d = 0; d < src.depth(); d++) {
        memcpy(buf_ + d * kKernelWidth, src.data(0, d), src.width());
      }
    }

    // Since the KernelInputScalar type may not be uint8, we need to cast buf_.
    complete_src_ = SrcMapType(reinterpret_cast<KernelInputScalar*>(buf_),
                               kKernelWidth, kRegisterSize);
  }
  // Packs a complete block into the destination. This is the most
  // critical part and the part that we most typically want to
  // override in architecture-specific optimized specializations.
  void Pack(PackedSideBlock* dst, int start_width) {
    std::uint8_t* dst_ptr = dst->current_data();
    for (int cell_start_depth = 0; cell_start_depth < kRegisterSize;
         cell_start_depth += kCellDepth) {
      for (int cell_start_width = 0; cell_start_width < kKernelWidth;
           cell_start_width += kCellWidth) {
        std::int32_t* cell_sums_of_each_slice_ptr =
            dst->sums_of_each_slice() + start_width + cell_start_width;
        const SideMap<const std::uint8_t, kSrcOrder> src_cell_map(
            complete_src_.block(cell_start_width, cell_start_depth, kCellWidth,
                                kCellDepth));
        for (int w = 0; w < kCellWidth; w++) {
          std::int32_t sum = 0;
          for (int d = 0; d < kCellDepth; d++) {
            const std::uint8_t src_val = src_cell_map(w, d);
            const std::int16_t kernel_val_unwrapped =
                src_val - kZeroPointInputValue;
            const std::uint8_t kernel_val_uint8 = kernel_val_unwrapped;
            dst_ptr[OffsetIntoCell<CellFormat>(w, d)] = kernel_val_uint8;
            sum += kernel_val_unwrapped;
          }
          cell_sums_of_each_slice_ptr[w] += sum;
        }
        dst_ptr += kCellSize;
      }
    }
    dst->seek_forward_n_cells(kCells * kRegisterSize / kCellDepth);
  }
};

template <typename SrcMapType, typename PackedSideBlock>
class PackingRegisterBlock
    : public PackingRegisterBlockBase<SrcMapType, PackedSideBlock> {};

// Large-scale implementation of packing.
template <typename SrcMapType, typename PackedSideBlock>
class PackSideBlockImpl {
 public:
  typedef typename PackedSideBlock::KernelSideFormat KernelSideFormat;
  typedef typename KernelSideFormat::Cell CellFormat;
  static constexpr int kCells = KernelSideFormat::kCells;
  static constexpr int kCellWidth = CellFormat::kWidth;
  static constexpr int kKernelWidth = CellFormat::kWidth * kCells;
  static constexpr int kCellDepth = CellFormat::kDepth;

  typedef PackingRegisterBlock<SrcMapType, PackedSideBlock>
      PackingRegisterBlockType;

  PackSideBlockImpl(PackedSideBlock* packed_side_block,
                    const SrcMapType& src_map)
      : packed_side_block_(packed_side_block), src_map_(src_map) {}

  PackedSideBlock* packed_side_block() const { return packed_side_block_; }

  const SrcMapType& src_map() const { return src_map_; }

  // The public entry point to pack a block.
  void PackL2() {
    memset(packed_side_block_->sums_of_each_slice(), 0,
           sizeof(std::int32_t) * packed_side_block_->params().l2_width);
    for (int d = 0; d < src_map_.depth();
         d += packed_side_block_->params().l1_depth) {
      int ds = std::min<int>(packed_side_block_->params().l1_depth,
                             src_map_.depth() - d);

      for (int w = 0; w < src_map_.width();
           w += packed_side_block_->params().l1_width) {
        int ws = std::min<int>(packed_side_block_->params().l1_width,
                               src_map_.width() - w);

        PrefetchL1(w, ws, d, ds);
        PackL1(w, ws, d, ds);
      }
    }
  }

 protected:
  // The intermediate-level loops, between PackL2 and PackRun.
  void PackL1(int start_width, int width, int start_depth, int depth) {
    for (int w = 0; w < width; w += kKernelWidth) {
      int ws = std::min(+kKernelWidth, width - w);
      packed_side_block_->seek_run(start_width + w, start_depth);
      PackRun(start_width + w, ws, start_depth, depth);
    }
  }

  // Prefetches the data that will be read by PackL1.
  void PrefetchL1(int start_width, int width, int start_depth, int depth) {
    if (SrcMapType::kOrder == SideMapOrder::WidthMajor) {
      for (int d = 0; d < depth; d += kDefaultCacheLineSize) {
        for (int w = 0; w < width; w += 1) {
          Prefetch(src_map_.data(start_width + w, start_depth + d));
        }
      }
    } else {
      for (int d = 0; d < depth; d++) {
        for (int w = 0; w < width; w += kDefaultCacheLineSize) {
          Prefetch(src_map_.data(start_width + w, start_depth + d));
        }
      }
    }
  }

  // PackRun packs only a run i.e. is the inner loop in the depth dimension.
  void PackRun(int start_width, int width, int start_depth, int depth) {
    PackingRegisterBlockType b;
    if (width == kKernelWidth) {
      const int register_aligned_depth = RoundDown<kRegisterSize>(depth);
      if (register_aligned_depth) {
        for (int d = 0; d < register_aligned_depth; d += kRegisterSize) {
          b.UseCompleteSrcInPlace(src_map_.block(start_width, start_depth + d,
                                                 width, kRegisterSize));
          b.Pack(packed_side_block_, start_width);
        }
      }
      if (register_aligned_depth < depth) {
        b.MakeCompleteSrc(
            src_map_.block(start_width, start_depth + register_aligned_depth,
                           width, depth - register_aligned_depth));
        b.Pack(packed_side_block_, start_width);
      }
    } else {
      assert(width < kKernelWidth);
      for (int d = 0; d < depth; d += kRegisterSize) {
        const int ds = std::min(+kRegisterSize, depth - d);
        b.MakeCompleteSrc(
            src_map_.block(start_width, start_depth + d, width, ds));
        b.Pack(packed_side_block_, start_width);
      }
    }
  }

  // The PackedSideBlock being packed, i.e. the 'destination'.
  PackedSideBlock* const packed_side_block_;

  // A map on the block of the original matrix block being packed,
  // i.e. the 'source'.
  const SrcMapType& src_map_;
};

// Packs a block of the input LHS matrix, into a PackedSideBlock.
template <typename PackedSideBlock, typename MatrixMapType>
void PackLhs(PackedSideBlock* dst, const MatrixMapType& src) {
  ScopedProfilingLabel label("pack LHS");
  static const SideMapOrder kSideMapOrder =
      MatrixMapType::kOrder == MapOrder::RowMajor ? SideMapOrder::WidthMajor
                                                  : SideMapOrder::DepthMajor;
  typedef typename MatrixMapType::Scalar Scalar;
  typedef SideMap<Scalar, kSideMapOrder> SideMapType;
  SideMapType src_side_map(src.data(), src.rows(), src.cols(), src.stride());
  typedef PackSideBlockImpl<SideMapType, PackedSideBlock> ImplType;
  ImplType impl(dst, src_side_map);
  impl.PackL2();
}

// Packs a block of the input RHS matrix, into a PackedSideBlock.
template <typename PackedSideBlock, typename MatrixMapType>
void PackRhs(PackedSideBlock* dst, const MatrixMapType& src) {
  ScopedProfilingLabel label("pack RHS");
  static const SideMapOrder kSideMapOrder =
      MatrixMapType::kOrder == MapOrder::ColMajor ? SideMapOrder::WidthMajor
                                                  : SideMapOrder::DepthMajor;
  typedef typename MatrixMapType::Scalar Scalar;
  typedef SideMap<Scalar, kSideMapOrder> SideMapType;
  SideMapType src_side_map(src.data(), src.cols(), src.rows(), src.stride());
  typedef PackSideBlockImpl<SideMapType, PackedSideBlock> ImplType;
  ImplType impl(dst, src_side_map);
  impl.PackL2();
}

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "pack_neon.h"
#elif defined(GEMMLOWP_SSE4)
#include "pack_sse.h"
#elif defined(GEMMLOWP_AVX2)
#include "pack_avx.h"
#elif defined(GEMMLOWP_MSA)
#include "pack_msa.h"
#endif

#endif  // GEMMLOWP_INTERNAL_PACK_H_
