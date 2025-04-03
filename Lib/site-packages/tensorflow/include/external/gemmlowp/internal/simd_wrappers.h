// Copyright 2017 The Gemmlowp Authors. All Rights Reserved.
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

// simd_wrappers.h: some inline functions wrapping SIMD intrinsics,
// extending the set of such functions from fixedpoint.h.

#ifndef GEMMLOWP_INTERNAL_SIMD_WRAPPERS_H_
#define GEMMLOWP_INTERNAL_SIMD_WRAPPERS_H_

#include <algorithm>
#include <type_traits>
#include "../fixedpoint/fixedpoint.h"

namespace gemmlowp {

template <typename ScalarType, int ScalarCount>
struct RegisterType {
  using Type = ScalarType;
};

inline std::int32_t Min(std::int32_t a, std::int32_t b) {
  return std::min(a, b);
}

inline std::int32_t Max(std::int32_t a, std::int32_t b) {
  return std::max(a, b);
}

inline void MulAdd(std::int32_t lhs, std::int32_t rhs, std::int32_t* acc) {
  *acc += lhs * rhs;
}

template <typename tScalarType, int tScalarCount>
struct RegisterBuffer {
  using ScalarType = tScalarType;
  static constexpr int kScalarCount = tScalarCount;
  using RegisterType = typename RegisterType<ScalarType, kScalarCount>::Type;
  static_assert((kScalarCount & (kScalarCount - 1)) == 0,
                "kScalarCount must be a power of two");
  static_assert(sizeof(RegisterType) % sizeof(ScalarType) == 0, "");
  static constexpr int kRegisterLanes =
      sizeof(RegisterType) / sizeof(ScalarType);
  static constexpr int kRegisterCount =
      (kScalarCount * sizeof(ScalarType) + sizeof(RegisterType) - 1) /
      sizeof(RegisterType);

  RegisterType reg[kRegisterCount];
};

template <typename tScalarType, int tRows, int tCols>
struct RegisterBlock {
  using ScalarType = tScalarType;
  static constexpr int kRows = tRows;
  static constexpr int kCols = tCols;
  static constexpr int kScalarCount = kRows * kCols;
  using BufferType = RegisterBuffer<ScalarType, kScalarCount>;
  using RegisterType = typename BufferType::RegisterType;
  static constexpr int kRegisterCount = BufferType::kRegisterCount;
  static constexpr int kRegisterLanes = BufferType::kRegisterLanes;

  BufferType buf;
};

template <typename RegisterBlockType>
struct RegisterBlockAddImpl {
  static RegisterBlockType Run(const RegisterBlockType& lhs,
                               const RegisterBlockType& rhs) {
    RegisterBlockType result;
    for (int i = 0; i < RegisterBlockType::kRegisterCount; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

template <typename RegisterBlockType>
RegisterBlockType RegisterBlockAdd(const RegisterBlockType& lhs,
                                   const RegisterBlockType& rhs) {
  return RegisterBlockAddImpl<RegisterBlockType>::Run(lhs, rhs);
}

template <typename LhsType, typename RhsType>
struct ShouldFlipLhsRhs {
  static constexpr bool kValue =
      (LhsType::kScalarCount < RhsType::kScalarCount) ||
      (LhsType::kScalarCount == RhsType::kScalarCount &&
       (LhsType::kRows < RhsType::kRows));
};

template <typename LhsType, typename RhsType,
          bool Flip = ShouldFlipLhsRhs<LhsType, RhsType>::kValue>
struct FlipLhsRhs {
  using FlippedLhsType = LhsType;
  using FlippedRhsType = RhsType;
  static const FlippedLhsType& FlippedLhs(const LhsType& lhs,
                                          const RhsType& rhs) {
    (void)rhs;
    return lhs;
  }
  static const FlippedRhsType& FlippedRhs(const LhsType& lhs,
                                          const RhsType& rhs) {
    (void)lhs;
    return rhs;
  }
};

template <typename LhsType, typename RhsType>
struct FlipLhsRhs<LhsType, RhsType, true> {
  using FlippedLhsType = RhsType;
  using FlippedRhsType = LhsType;
  static const FlippedLhsType& FlippedLhs(const LhsType& lhs,
                                          const RhsType& rhs) {
    (void)lhs;
    return rhs;
  }
  static const FlippedRhsType& FlippedRhs(const LhsType& lhs,
                                          const RhsType& rhs) {
    (void)rhs;
    return lhs;
  }
};

template <typename Lhs, typename Rhs>
struct BroadcastBinaryOpShape {
  static constexpr int kRows =
      Lhs::kRows > Rhs::kRows ? Lhs::kRows : Rhs::kRows;
  static constexpr int kCols =
      Lhs::kCols > Rhs::kCols ? Lhs::kCols : Rhs::kCols;
};

template <typename Lhs, typename Rhs>
struct BroadcastBinaryOpRegisterBlock {
  using Shape = BroadcastBinaryOpShape<Lhs, Rhs>;
  using ScalarType = typename Lhs::ScalarType;
  using Type = RegisterBlock<ScalarType, Shape::kRows, Shape::kCols>;
};

template <typename Lhs, typename Rhs>
struct BroadcastAddImpl {
  using ResultBlockType =
      typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type;
  static ResultBlockType Run(const Lhs& lhs, const Rhs& rhs) {
    ResultBlockType result;
    static constexpr int Rows = ResultBlockType::kRows;
    static constexpr int Cols = ResultBlockType::kCols;
    static constexpr int LhsRows = Lhs::kRows;
    static constexpr int LhsCols = Lhs::kCols;
    static constexpr int RhsRows = Rhs::kRows;
    static constexpr int RhsCols = Rhs::kCols;

    static_assert(LhsRows == Rows || LhsRows == 1, "");
    static_assert(RhsRows == Rows || RhsRows == 1, "");
    static_assert(LhsCols == Cols || LhsCols == 1, "");
    static_assert(RhsCols == Cols || RhsCols == 1, "");
    static_assert(ResultBlockType::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Lhs::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Rhs::kRegisterLanes == 1,
                  "This path is only for scalar values");

    for (int c = 0; c < Cols; c++) {
      const int lhs_c = LhsCols == Cols ? c : 0;
      const int rhs_c = RhsCols == Cols ? c : 0;
      for (int r = 0; r < Rows; r++) {
        const int lhs_r = LhsRows == Rows ? r : 0;
        const int rhs_r = RhsRows == Rows ? r : 0;
        result.buf.reg[r + c * Rows] =
            Add(lhs.buf.reg[lhs_r + lhs_c * LhsRows],
                rhs.buf.reg[rhs_r + rhs_c * RhsRows]);
      }
    }
    return result;
  }
};

template <typename Lhs, typename Rhs>
typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type BroadcastAdd(
    const Lhs& lhs, const Rhs& rhs) {
  using Flip = FlipLhsRhs<Lhs, Rhs>;
  return BroadcastAddImpl<
      typename Flip::FlippedLhsType,
      typename Flip::FlippedRhsType>::Run(Flip::FlippedLhs(lhs, rhs),
                                          Flip::FlippedRhs(lhs, rhs));
}

template <typename Lhs, typename Rhs>
struct BroadcastShiftLeftImpl {
  using ResultBlockType =
      typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type;
  static ResultBlockType Run(const Lhs& lhs, const Rhs& rhs) {
    ResultBlockType result;
    static constexpr int Rows = ResultBlockType::kRows;
    static constexpr int Cols = ResultBlockType::kCols;
    static constexpr int LhsRows = Lhs::kRows;
    static constexpr int LhsCols = Lhs::kCols;
    static constexpr int RhsRows = Rhs::kRows;
    static constexpr int RhsCols = Rhs::kCols;

    static_assert(LhsRows == Rows || LhsRows == 1, "");
    static_assert(RhsRows == Rows || RhsRows == 1, "");
    static_assert(LhsCols == Cols || LhsCols == 1, "");
    static_assert(RhsCols == Cols || RhsCols == 1, "");
    static_assert(ResultBlockType::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Lhs::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Rhs::kRegisterLanes == 1,
                  "This path is only for scalar values");

    for (int c = 0; c < Cols; c++) {
      const int lhs_c = LhsCols == Cols ? c : 0;
      const int rhs_c = RhsCols == Cols ? c : 0;
      for (int r = 0; r < Rows; r++) {
        const int lhs_r = LhsRows == Rows ? r : 0;
        const int rhs_r = RhsRows == Rows ? r : 0;
        result.buf.reg[r + c * Rows] =
            ShiftLeft(lhs.buf.reg[lhs_r + lhs_c * LhsRows],
                      rhs.buf.reg[rhs_r + rhs_c * RhsRows]);
      }
    }
    return result;
  }
};

template <typename Lhs, typename Rhs>
typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type BroadcastShiftLeft(
    const Lhs& lhs, const Rhs& rhs) {
  using Flip = FlipLhsRhs<Lhs, Rhs>;
  return BroadcastShiftLeftImpl<
      typename Flip::FlippedLhsType,
      typename Flip::FlippedRhsType>::Run(Flip::FlippedLhs(lhs, rhs),
                                          Flip::FlippedRhs(lhs, rhs));
}

template <typename Lhs, typename Rhs>
struct BroadcastSaturatingRoundingDoublingHighMulImpl {
  using ResultBlockType =
      typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type;
  static ResultBlockType Run(const Lhs& lhs, const Rhs& rhs) {
    ResultBlockType result;
    static constexpr int Rows = ResultBlockType::kRows;
    static constexpr int Cols = ResultBlockType::kCols;
    static constexpr int LhsRows = Lhs::kRows;
    static constexpr int LhsCols = Lhs::kCols;
    static constexpr int RhsRows = Rhs::kRows;
    static constexpr int RhsCols = Rhs::kCols;

    static_assert(LhsRows == Rows || LhsRows == 1, "");
    static_assert(RhsRows == Rows || RhsRows == 1, "");
    static_assert(LhsCols == Cols || LhsCols == 1, "");
    static_assert(RhsCols == Cols || RhsCols == 1, "");
    static_assert(ResultBlockType::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Lhs::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Rhs::kRegisterLanes == 1,
                  "This path is only for scalar values");

    for (int c = 0; c < Cols; c++) {
      const int lhs_c = LhsCols == Cols ? c : 0;
      const int rhs_c = RhsCols == Cols ? c : 0;
      for (int r = 0; r < Rows; r++) {
        const int lhs_r = LhsRows == Rows ? r : 0;
        const int rhs_r = RhsRows == Rows ? r : 0;
        result.buf.reg[r + c * Rows] = SaturatingRoundingDoublingHighMul(
            lhs.buf.reg[lhs_r + lhs_c * LhsRows],
            rhs.buf.reg[rhs_r + rhs_c * RhsRows]);
      }
    }
    return result;
  }
};

template <typename Lhs, typename Rhs>
typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type
BroadcastSaturatingRoundingDoublingHighMul(const Lhs& lhs, const Rhs& rhs) {
  using Flip = FlipLhsRhs<Lhs, Rhs>;
  return BroadcastSaturatingRoundingDoublingHighMulImpl<
      typename Flip::FlippedLhsType,
      typename Flip::FlippedRhsType>::Run(Flip::FlippedLhs(lhs, rhs),
                                          Flip::FlippedRhs(lhs, rhs));
}

template <typename Lhs, typename Rhs>
struct BroadcastRoundingDivideByPOTImpl {
  using ResultBlockType =
      typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type;
  static ResultBlockType Run(const Lhs& lhs, const Rhs& rhs) {
    ResultBlockType result;
    static constexpr int Rows = ResultBlockType::kRows;
    static constexpr int Cols = ResultBlockType::kCols;
    static constexpr int LhsRows = Lhs::kRows;
    static constexpr int LhsCols = Lhs::kCols;
    static constexpr int RhsRows = Rhs::kRows;
    static constexpr int RhsCols = Rhs::kCols;

    static_assert(LhsRows == Rows || LhsRows == 1, "");
    static_assert(RhsRows == Rows || RhsRows == 1, "");
    static_assert(LhsCols == Cols || LhsCols == 1, "");
    static_assert(RhsCols == Cols || RhsCols == 1, "");
    static_assert(ResultBlockType::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Lhs::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Rhs::kRegisterLanes == 1,
                  "This path is only for scalar values");

    for (int c = 0; c < Cols; c++) {
      const int lhs_c = LhsCols == Cols ? c : 0;
      const int rhs_c = RhsCols == Cols ? c : 0;
      for (int r = 0; r < Rows; r++) {
        const int lhs_r = LhsRows == Rows ? r : 0;
        const int rhs_r = RhsRows == Rows ? r : 0;
        result.buf.reg[r + c * Rows] =
            RoundingDivideByPOT(lhs.buf.reg[lhs_r + lhs_c * LhsRows],
                                rhs.buf.reg[rhs_r + rhs_c * RhsRows]);
      }
    }
    return result;
  }
};

template <typename Lhs, typename Rhs>
typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type
BroadcastRoundingDivideByPOT(const Lhs& lhs, const Rhs& rhs) {
  using Flip = FlipLhsRhs<Lhs, Rhs>;
  return BroadcastRoundingDivideByPOTImpl<
      typename Flip::FlippedLhsType,
      typename Flip::FlippedRhsType>::Run(Flip::FlippedLhs(lhs, rhs),
                                          Flip::FlippedRhs(lhs, rhs));
}

template <typename Lhs, typename Rhs>
struct BroadcastMulImpl {
  using ResultBlockType =
      typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type;
  static ResultBlockType Run(const Lhs& lhs, const Rhs& rhs) {
    ResultBlockType result;
    static constexpr int Rows = ResultBlockType::kRows;
    static constexpr int Cols = ResultBlockType::kCols;
    static constexpr int LhsRows = Lhs::kRows;
    static constexpr int LhsCols = Lhs::kCols;
    static constexpr int RhsRows = Rhs::kRows;
    static constexpr int RhsCols = Rhs::kCols;
    static_assert(ResultBlockType::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Lhs::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Rhs::kRegisterLanes == 1,
                  "This path is only for scalar values");

    static_assert(LhsRows == Rows || LhsRows == 1, "");
    static_assert(RhsRows == Rows || RhsRows == 1, "");
    static_assert(LhsCols == Cols || LhsCols == 1, "");
    static_assert(RhsCols == Cols || RhsCols == 1, "");
    for (int c = 0; c < Cols; c++) {
      const int lhs_c = LhsCols == Cols ? c : 0;
      const int rhs_c = RhsCols == Cols ? c : 0;
      for (int r = 0; r < Rows; r++) {
        const int lhs_r = LhsRows == Rows ? r : 0;
        const int rhs_r = RhsRows == Rows ? r : 0;
        result.buf.reg[r + c * Rows] =
            Mul(lhs.buf.reg[lhs_r + lhs_c * LhsRows],
                rhs.buf.reg[rhs_r + rhs_c * RhsRows]);
      }
    }
    return result;
  }
};

template <typename Lhs, typename Rhs>
typename BroadcastBinaryOpRegisterBlock<Lhs, Rhs>::Type BroadcastMul(
    const Lhs& lhs, const Rhs& rhs) {
  using Flip = FlipLhsRhs<Lhs, Rhs>;
  return BroadcastMulImpl<
      typename Flip::FlippedLhsType,
      typename Flip::FlippedRhsType>::Run(Flip::FlippedLhs(lhs, rhs),
                                          Flip::FlippedRhs(lhs, rhs));
}

template <typename Lhs, typename Rhs, typename Acc>
struct BroadcastMulAddImpl {
  static void Run(const Lhs& lhs, const Rhs& rhs, Acc* acc) {
    static constexpr int Rows = Acc::kRows;
    static constexpr int Cols = Acc::kCols;
    static constexpr int LhsRows = Lhs::kRows;
    static constexpr int LhsCols = Lhs::kCols;
    static constexpr int RhsRows = Rhs::kRows;
    static constexpr int RhsCols = Rhs::kCols;
    static_assert(Acc::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Lhs::kRegisterLanes == 1,
                  "This path is only for scalar values");
    static_assert(Rhs::kRegisterLanes == 1,
                  "This path is only for scalar values");

    static_assert(LhsRows == Rows || LhsRows == 1, "");
    static_assert(RhsRows == Rows || RhsRows == 1, "");
    static_assert(LhsCols == Cols || LhsCols == 1, "");
    static_assert(RhsCols == Cols || RhsCols == 1, "");
    for (int c = 0; c < Cols; c++) {
      const int lhs_c = LhsCols == Cols ? c : 0;
      const int rhs_c = RhsCols == Cols ? c : 0;
      for (int r = 0; r < Rows; r++) {
        const int lhs_r = LhsRows == Rows ? r : 0;
        const int rhs_r = RhsRows == Rows ? r : 0;
        MulAdd(lhs.buf.reg[lhs_r + lhs_c * LhsRows],
               rhs.buf.reg[rhs_r + rhs_c * RhsRows],
               &acc->buf.reg[r + c * Rows]);
      }
    }
  }
};

template <typename Lhs, typename Rhs, typename Acc>
void BroadcastMulAdd(const Lhs& lhs, const Rhs& rhs, Acc* acc) {
  using Flip = FlipLhsRhs<Lhs, Rhs>;
  BroadcastMulAddImpl<typename Flip::FlippedLhsType,
                      typename Flip::FlippedRhsType,
                      Acc>::Run(Flip::FlippedLhs(lhs, rhs),
                                Flip::FlippedRhs(lhs, rhs), acc);
}

template <typename RegisterBlockType, typename SrcObjectType>
struct LoadImpl {
  static_assert(std::is_same<SrcObjectType, void>::value,
                "This generic impl should never be hit");
};

template <typename ScalarType, int Rows, int Cols, typename SrcScalarType>
struct LoadImpl<RegisterBlock<ScalarType, Rows, Cols>,
                MatrixMap<SrcScalarType, MapOrder::ColMajor>> {
  using RegisterBlockType = RegisterBlock<ScalarType, Rows, Cols>;
  using SrcObjectType = MatrixMap<SrcScalarType, MapOrder::ColMajor>;
  static RegisterBlockType Run(const SrcObjectType& src, int row, int col) {
    RegisterBlockType result;
    int i = 0;
    for (int c = 0; c < Cols; c++) {
      const ScalarType* src_ptr = src.data(row, col + c);
      for (int r = 0; r < Rows; r++) {
        result.buf.reg[i++] = *src_ptr++;
      }
    }
    return result;
  }
};

template <typename ScalarType, int Rows, int Cols, typename SrcScalarType,
          VectorShape Shape>
struct LoadImpl<RegisterBlock<ScalarType, Rows, Cols>,
                VectorMap<SrcScalarType, Shape>> {
  using RegisterBlockType = RegisterBlock<ScalarType, Rows, Cols>;
  using SrcObjectType = VectorMap<SrcScalarType, Shape>;
  static RegisterBlockType Run(const SrcObjectType& src, int pos) {
    static_assert(Shape == VectorShape::Col || Rows == 1, "");
    static_assert(Shape == VectorShape::Row || Cols == 1, "");
    RegisterBlockType result;
    for (int i = 0; i < Rows * Cols; i++) {
      result.buf.reg[i] = src(pos + i);
    }
    return result;
  }
};

template <typename ScalarType, int Rows, int Cols, typename SrcScalarType,
          VectorShape Shape>
struct LoadImpl<RegisterBlock<ScalarType, Rows, Cols>,
                VectorDup<SrcScalarType, Shape>> {
  using RegisterBlockType = RegisterBlock<ScalarType, Rows, Cols>;
  using SrcObjectType = VectorDup<SrcScalarType, Shape>;
  static RegisterBlockType Run(const SrcObjectType& src, int) {
    static_assert(Shape == VectorShape::Col || Rows == 1, "");
    static_assert(Shape == VectorShape::Row || Cols == 1, "");
    RegisterBlockType result;
    for (int i = 0; i < Rows * Cols; i++) {
      result.buf.reg[i] = src(0);
    }
    return result;
  }
};

template <typename RegisterBlockType, typename SrcObjectType>
RegisterBlockType Load(const SrcObjectType& src, int row, int col) {
  return LoadImpl<RegisterBlockType, SrcObjectType>::Run(src, row, col);
}

template <typename RegisterBlockType, typename SrcObjectType>
RegisterBlockType Load(const SrcObjectType& src, int pos) {
  return LoadImpl<RegisterBlockType, SrcObjectType>::Run(src, pos);
}

template <typename RegisterBlockType>
struct LoadContiguousImpl {
  using ScalarType = typename RegisterBlockType::ScalarType;
  static_assert(RegisterBlockType::kRegisterLanes == 1,
                "This path is only for scalar values");
  static RegisterBlockType Run(const ScalarType* src) {
    RegisterBlockType result;
    for (int i = 0; i < RegisterBlockType::kScalarCount; i++) {
      result.buf.reg[i] = src[i];
    }
    return result;
  }
};

template <typename RegisterBlockType>
RegisterBlockType LoadContiguous(
    const typename RegisterBlockType::ScalarType* src) {
  return LoadContiguousImpl<RegisterBlockType>::Run(src);
}

template <int BroadcastRows, int BroadcastCols, typename SrcObjectType>
struct LoadForBroadcastingShape {};

template <int BroadcastRows, int BroadcastCols, typename ScalarType,
          VectorShape Shape>
struct LoadForBroadcastingShape<BroadcastRows, BroadcastCols,
                                VectorMap<ScalarType, Shape>> {
  static constexpr int kRows = Shape == VectorShape::Col ? BroadcastRows : 1;
  static constexpr int kCols = Shape == VectorShape::Row ? BroadcastCols : 1;
};

template <int BroadcastRows, int BroadcastCols, typename ScalarType,
          VectorShape Shape>
struct LoadForBroadcastingShape<BroadcastRows, BroadcastCols,
                                VectorDup<ScalarType, Shape>> {
  static constexpr int kRows = 1;
  static constexpr int kCols = 1;
};

template <typename RegisterBlockType, typename SrcObjectType>
struct LoadForBroadcastingRegisterBlock {
  using Shape =
      LoadForBroadcastingShape<RegisterBlockType::kRows,
                               RegisterBlockType::kCols, SrcObjectType>;
  using ScalarType = typename RegisterBlockType::ScalarType;
  using Type = RegisterBlock<ScalarType, Shape::kRows, Shape::kCols>;
};

template <typename RegisterBlockType, typename SrcObjectType>
struct LoadForBroadcastingImpl {
  static_assert(std::is_same<SrcObjectType, void>::value,
                "This generic impl should never be hit");
};

template <typename ScalarType, int Rows, int Cols, typename SrcScalarType,
          VectorShape Shape>
struct LoadForBroadcastingImpl<RegisterBlock<ScalarType, Rows, Cols>,
                               VectorMap<SrcScalarType, Shape>> {
  using RegisterBlockType = RegisterBlock<ScalarType, Rows, Cols>;
  using SrcObjectType = VectorMap<SrcScalarType, Shape>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;
  static_assert(ResultBlockType::kRegisterLanes == 1,
                "This path is only for scalar values");
  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    for (int c = 0; c < ResultBlockType::kCols; c++) {
      for (int r = 0; r < ResultBlockType::kRows; r++) {
        const int i = Shape == VectorShape::Col ? r : c;
        result.buf.reg[r + c * ResultBlockType::kRows] = src(pos + i);
      }
    }
    return result;
  }
};

template <typename ScalarType, int Rows, int Cols, typename SrcScalarType,
          VectorShape Shape>
struct LoadForBroadcastingImpl<RegisterBlock<ScalarType, Rows, Cols>,
                               VectorDup<SrcScalarType, Shape>> {
  using RegisterBlockType = RegisterBlock<ScalarType, Rows, Cols>;
  using SrcObjectType = VectorDup<SrcScalarType, Shape>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;
  static_assert(ResultBlockType::kRegisterLanes == 1,
                "This path is only for scalar values");
  static ResultBlockType Run(const SrcObjectType& src, int) {
    ResultBlockType result;
    for (int c = 0; c < ResultBlockType::kCols; c++) {
      for (int r = 0; r < ResultBlockType::kRows; r++) {
        result.buf.reg[r + c * ResultBlockType::kRows] = src(0);
      }
    }
    return result;
  }
};

template <typename RegisterBlockType, typename SrcObjectType>
typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                          SrcObjectType>::Type
LoadForBroadcasting(const SrcObjectType& src, int row, int col) {
  return LoadForBroadcastingImpl<RegisterBlockType, SrcObjectType>::Run(
      src, row, col);
}

template <typename RegisterBlockType, typename SrcObjectType>
typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                          SrcObjectType>::Type
LoadForBroadcasting(const SrcObjectType& src, int pos) {
  return LoadForBroadcastingImpl<RegisterBlockType, SrcObjectType>::Run(src,
                                                                        pos);
}

template <int ConstantValue, typename RegisterBlockType>
struct AddConstantImpl {
  static void Run(RegisterBlockType* block) {
    using RegisterType = typename RegisterBlockType::RegisterType;
    const RegisterType dup = Dup<RegisterType>(ConstantValue);
    for (int i = 0; i < RegisterBlockType::kRegisterCount; i++) {
      block->buf.reg[i] = Add(block->buf.reg[i], dup);
    }
  }
};

template <typename RegisterBlockType>
struct AddConstantImpl<0, RegisterBlockType> {
  static void Run(RegisterBlockType*) {
    // This is a no-op.
  }
};

template <int ConstantValue, typename RegisterBlockType>
void AddConstant(RegisterBlockType* block) {
  AddConstantImpl<ConstantValue, RegisterBlockType>::Run(block);
}

template <int N>
using RegBufferInt32 = RegisterBuffer<std::int32_t, N>;
template <int N>
using RegBufferInt16 = RegisterBuffer<std::int16_t, N>;
template <int N>
using RegBufferUint8 = RegisterBuffer<std::uint8_t, N>;
template <int N>
using RegBufferInt8 = RegisterBuffer<std::int8_t, N>;
template <int R, int C>
using RegBlockInt32 = RegisterBlock<std::int32_t, R, C>;
template <int R, int C>
using RegBlockInt16 = RegisterBlock<std::int16_t, R, C>;
template <int R, int C>
using RegBlockUint8 = RegisterBlock<std::uint8_t, R, C>;
template <int R, int C>
using RegBlockInt8 = RegisterBlock<std::int8_t, R, C>;

}  // end namespace gemmlowp

#if defined GEMMLOWP_NEON
#include "simd_wrappers_neon.h"
#elif defined GEMMLOWP_SSE4
#include "simd_wrappers_sse.h"
#elif defined GEMMLOWP_MSA
#include "simd_wrappers_msa.h"
#endif

#endif  // GEMMLOWP_INTERNAL_SIMD_WRAPPERS_H_
