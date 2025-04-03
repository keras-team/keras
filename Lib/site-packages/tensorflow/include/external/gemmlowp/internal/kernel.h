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

// kernel.h: general definitions for kernels.

#ifndef GEMMLOWP_INTERNAL_KERNEL_H_
#define GEMMLOWP_INTERNAL_KERNEL_H_

#include "../public/bit_depth.h"
#include "common.h"

namespace gemmlowp {

// Explanation of general gemmlowp terminology
// ===========================================
//
// We use the following abbreviations:
// LHS = "left-hand side"
// RHS = "right-hand side"
// Sometimes when referring to either LHS or RHS, we just say a "Side".
//
// In a matrix product of a MxK matrix times a KxN matrix,
// we call K the 'depth'. Note that M is the number of rows
// of the result (and of the LHS), and N is the number of columns
// of the result (and of the RHS).
//
// In each of the LHS and RHS matrices, we call 'width' the
// other dimension, besides the depth. So in the LHS, 'width'
// is the number of rows, while in the RHS, 'width' is the number
// of columns.
//
//  So in the LHS MxK matrix, the depth is K and the width in M.
// And in the RHS KxN matrix, the depth is K and the width in N.
//
// This is illustrated in this picture:
//
//                             RHS width
//                        <----------------->
//                        +-----------------+ ^
//                        |       RHS       | | Depth
//                        +-----------------+ v
//                 ^ +--+ +-----------------+
//                 | |L | |                 |
//       LHS width | |H | |      Result     |
//                 | |S | |                 |
//                 v +--+ +-----------------+
//                   <-->
//                   Depth

// Explanation of gemmlowp kernel formats and "cells"
// ==================================================
//
// Kernels operate on small LHS and RHS blocks that fit in registers.
// These blocks are stored contiguously in memory, but not always
// in a traditional column-major or row-major order; instead,
// they consist of a number of sub-blocks, which we call "cells",
// that are stored in column-major or row-major order. However,
// what really matters to us is not so much rows vs columns, but
// rather width vs depth. So we refer to "width-major" and "depth-major"
// storage orders. In the LHS, width-major means row-major,
// while in the RHS, width-major means column-major.
// There is also a third possibility, "diagonal order",
// which is unused at the moment.
//
// We aim to treat both sides, LHS and RHS, on an equal footing,
// so we call them both 'sides'. A KernelFormat thus is just a pair
// of KernelSideFormat's, one for LHS and one for RHS; each KernelSideFormat
// contains a CellFormat and a number of cells; cells are only ever
// stacked in the width dimension, which means stacked vertically in the
// LHS and stacked horizondally in the RHS.
//
// Example
// =======
//
// Let's work out the data layout expected by a kernel having the
// following format (the struct names here are defined below in this file):
//
// KernelFormat<
//   KernelSideFormat<CellFormat<3, 4>, 3>,
//   KernelSideFormat<CellFormat<5, 4>, 2>
// >
//
// The LHS format, KernelSideFormat<CellFormat<3, 4>, 3>, means:
// 3 cells, each cell having dimensions (width=3, depth=4), laid out in
// DepthMajor order (the default value, see CellFormat). In the LHS,
// DepthMajor means column-major, so the LHS cells are of size 3x4 in
// column-major order, so the LHS layout is:
//
// 0  3  6  9
// 1  4  7  10
// 2  5  8  11
// 12 15 18 21
// 13 16 19 22
// 14 17 20 23
// 24 27 30 33
// 25 28 31 34
// 26 29 32 35
//
// The RHS format, KernelSideFormat<CellFormat<5, 4>, 2>, means:
// 2 cells each having dimensions (width=5, depth=4), laid out in
// DepthMajor order (the default value, see CellFormat). In the RHS,
// DepthMajor means row-major, so the RHS cells are of size 4x5 in
// row-major order, so the RHS layout is:
//
// 0  1  2  3  4  20 21 22 23 24
// 5  6  7  8  9  25 26 27 28 29
// 10 11 12 13 14 30 31 32 33 34
// 15 16 17 18 19 35 36 37 38 39

// CellOrder enumerates the possible storage orders (=layouts) for
// a cell (see explanation above).
enum class CellOrder { DepthMajor, WidthMajor, Diagonal };

// CellFormat describes how data is laid
// out in a cell. That is, a CellOrder together with actual dimensions.
template <int tWidth, int tDepth, CellOrder tOrder = CellOrder::DepthMajor>
struct CellFormat {
  static constexpr int kWidth = tWidth;
  static constexpr int kDepth = tDepth;
  static constexpr CellOrder kOrder = tOrder;

  static constexpr int kSize = kWidth * kDepth;
};

// KernelSideFormat describes how data is laid out in a kernel side
// (i.e. LHS or RHS). That is, a CellFormat together with a number of
// cells. These cells are always stacked in the Width dimension.
// For example, in the LHS case, the Width dimension is the rows dimension,
// se we're saying that in the LHS, cells are stacked vertically.
// We never stack cells in the Depth dimension.
template <typename tCellFormat, int tCells>
struct KernelSideFormat {
  typedef tCellFormat Cell;
  static constexpr int kCells = tCells;
  static constexpr int kWidth = kCells * Cell::kWidth;
  static constexpr int kDepth = Cell::kDepth;
  typedef std::uint8_t Scalar;       // The scalar type of the Format.
  typedef std::uint8_t InputScalar;  // The scalar type of the original input.
};

// KernelSideFormat for int8 fast kernel trick. The original input is uint8, but
// packs converts it to int8.
template <typename tCellFormat, int tCells>
struct KernelSideFormatInt8 : KernelSideFormat<tCellFormat, tCells> {
  typedef std::int8_t Scalar;
  typedef std::uint8_t InputScalar;
};

// KernelSideFormat for int8 inputs, enabling int8 fast kernel trick without
// pack conversion.
template <typename tCellFormat, int tCells>
struct KernelSideFormatInt8Inputs : KernelSideFormat<tCellFormat, tCells> {
  typedef std::int8_t Scalar;
  typedef std::int8_t InputScalar;
};

// KernelFormat describes fully the input data layout that a kernel expects.
// It consists of two KernelSideFormat's, one for LHS and one for RHS.
template <typename tLhs, typename tRhs>
struct KernelFormat {
  typedef tLhs Lhs;
  typedef tRhs Rhs;

  static_assert(Lhs::Cell::kDepth == Rhs::Cell::kDepth, "");
  static constexpr int kDepth = Lhs::Cell::kDepth;
  static constexpr int kRows = Lhs::Cell::kWidth * Lhs::kCells;
  static constexpr int kCols = Rhs::Cell::kWidth * Rhs::kCells;
};

inline const char* CellOrderName(CellOrder o) {
  switch (o) {
    case CellOrder::DepthMajor:
      return "DepthMajor";
    case CellOrder::WidthMajor:
      return "WidthMajor";
    case CellOrder::Diagonal:
      return "Diagonal";
    default:
      assert(false);
      return nullptr;
  }
}

// Returns the offset into a cell, at which a given coefficient is stored.
template <typename CellFormat>
inline int OffsetIntoCell(int w, int d) {
  const int size = CellFormat::kWidth;
  switch (CellFormat::kOrder) {
    case CellOrder::DepthMajor:
      return w + d * CellFormat::kWidth;
    case CellOrder::WidthMajor:
      return d + w * CellFormat::kDepth;
    case CellOrder::Diagonal:
      assert(CellFormat::kWidth == CellFormat::kDepth);
      return ((size + w - d) * size + d) % (size * size);
    default:
      assert(false);
      return 0;
  }
}

// KernelBase is the virtual base class below all kernels.
// The idea is that we don't need to templatize all our code on the exact
// kernel type; we only need to templatize on kernel format. Kernels
// sharing the same format can thus share the same packing/unpacking code.
struct KernelBase {
  virtual const char* Name() const = 0;

  // This is the kernel implementation. We use the word 'run' consistently
  // throughout gemmlowp to mean an inner loop, the implementation of which
  // is to be provided by a separate optimized function.
  virtual void Run(std::int32_t* dst_ptr, std::size_t dst_row_stride,
                   std::size_t dst_col_stride, const std::uint8_t* lhs_ptr,
                   const std::uint8_t* rhs_ptr, std::size_t start_depth,
                   std::size_t run_depth) const = 0;

  virtual ~KernelBase() {}
};

template <typename InputKernelScalarType, typename KernelScalarType>
struct ZeroPointInputValue {};

template <>
struct ZeroPointInputValue<std::uint8_t, std::uint8_t> {
  static constexpr std::uint8_t kValue = 0;
};

template <>
struct ZeroPointInputValue<std::uint8_t, std::int8_t> {
  static constexpr std::uint8_t kValue = 128;
};

template <>
struct ZeroPointInputValue<std::int8_t, std::int8_t> {
  static constexpr std::uint8_t kValue = 0;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_KERNEL_H_
