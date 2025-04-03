// Copyright 2015 Google Inc. All Rights Reserved.
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

// simd_wrappers_common_neon_sse.h: common SIMD (NEON and SSE) wrapper code

#ifndef GEMMLOWP_INTERNAL_SIMD_WRAPPERS_COMMON_NEON_SSE_H_
#define GEMMLOWP_INTERNAL_SIMD_WRAPPERS_COMMON_NEON_SSE_H_

#include "simd_wrappers.h"

namespace gemmlowp {

template <typename SrcScalarType, int N>
struct LoadImpl<RegBlockInt32<4, N>,
                MatrixMap<SrcScalarType, MapOrder::ColMajor>> {
  static RegBlockInt32<4, N> Run(
      const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row,
      int col) {
    RegBlockInt32<4, N> result;
    for (int i = 0; i < N; i++) {
      result.buf.reg[i] = LoadInt32x4(src.data(row, col + i));
    }
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadImpl<RegBlockInt32<8, N>,
                MatrixMap<SrcScalarType, MapOrder::ColMajor>> {
  static RegBlockInt32<8, N> Run(
      const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row,
      int col) {
    RegBlockInt32<8, N> result;
    for (int i = 0; i < N; i++) {
      result.buf.reg[2 * i + 0] = LoadInt32x4(src.data(row + 0, col + i));
      result.buf.reg[2 * i + 1] = LoadInt32x4(src.data(row + 4, col + i));
    }
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<1, 4>,
                MatrixMap<SrcScalarType, MapOrder::ColMajor>> {
  static RegBlockInt32<1, 4> Run(
      const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row,
      int col) {
    RegBlockInt32<1, 4> result;
    std::int32_t buf[4];
    for (int i = 0; i < 4; i++) {
      buf[i] = src(row, col + i);
    }
    result.buf.reg[0] = LoadInt32x4(buf);
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<1, 8>,
                MatrixMap<SrcScalarType, MapOrder::ColMajor>> {
  static RegBlockInt32<1, 8> Run(
      const MatrixMap<SrcScalarType, MapOrder::ColMajor>& src, int row,
      int col) {
    RegBlockInt32<1, 8> result;
    std::int32_t buf[8];
    for (int i = 0; i < 8; i++) {
      buf[i] = src(row, col + i);
    }
    result.buf.reg[0] = LoadInt32x4(buf);
    result.buf.reg[1] = LoadInt32x4(buf + 4);
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<4, 1>,
                VectorMap<SrcScalarType, VectorShape::Col>> {
  static RegBlockInt32<4, 1> Run(
      const VectorMap<SrcScalarType, VectorShape::Col>& src, int pos) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    return result;
  }
};

template <typename SrcScalarType>
struct LoadImpl<RegBlockInt32<4, 1>,
                VectorDup<SrcScalarType, VectorShape::Col>> {
  static RegBlockInt32<4, 1> Run(
      const VectorDup<SrcScalarType, VectorShape::Col>& src, int) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = LoadInt32x4(src(0));
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<4, N>,
                               VectorMap<SrcScalarType, VectorShape::Col>> {
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Col>;
  using RegisterBlockType = RegBlockInt32<4, N>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 1, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<8, N>,
                               VectorMap<SrcScalarType, VectorShape::Col>> {
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Col>;
  using RegisterBlockType = RegBlockInt32<8, N>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 2, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    result.buf.reg[1] = LoadInt32x4(src.data(pos + 4));
    return result;
  }
};

template <typename SrcScalarType>
struct LoadForBroadcastingImpl<RegBlockInt32<4, 1>,
                               VectorMap<SrcScalarType, VectorShape::Row>> {
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Row>;
  using RegisterBlockType = RegBlockInt32<4, 1>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    result.buf.reg[0] = src(pos);
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<N, 4>,
                               VectorMap<SrcScalarType, VectorShape::Row>> {
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Row>;
  using RegisterBlockType = RegBlockInt32<N, 4>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 1, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    return result;
  }
};

template <typename SrcScalarType, int N>
struct LoadForBroadcastingImpl<RegBlockInt32<N, 8>,
                               VectorMap<SrcScalarType, VectorShape::Row>> {
  using SrcObjectType = VectorMap<SrcScalarType, VectorShape::Row>;
  using RegisterBlockType = RegBlockInt32<N, 8>;
  using ResultBlockType =
      typename LoadForBroadcastingRegisterBlock<RegisterBlockType,
                                                SrcObjectType>::Type;

  static ResultBlockType Run(const SrcObjectType& src, int pos) {
    ResultBlockType result;
    static_assert(ResultBlockType::kRegisterCount == 2, "");
    result.buf.reg[0] = LoadInt32x4(src.data(pos));
    result.buf.reg[1] = LoadInt32x4(src.data(pos + 4));
    return result;
  }
};

// 4x1 := 4x1 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4, 1>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 4x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4, 1>, RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 + 1x4
template <>
struct BroadcastAddImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 + 1x4
template <>
struct BroadcastAddImpl<RegBlockInt32<4, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[2] = Add(lhs.buf.reg[2], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[3] = Add(lhs.buf.reg[3], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x4 := 4x4 + 4x1
template <>
struct BroadcastAddImpl<RegBlockInt32<4, 4>, RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[0]);
    result.buf.reg[2] = Add(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = Add(lhs.buf.reg[3], rhs.buf.reg[0]);
    return result;
  }
};

// 8x1 := 8x1 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8, 1>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    const Int32x4 p = Dup<Int32x4>(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 + 8x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8, 1>, RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Add(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 + 1x4
template <>
struct BroadcastAddImpl<RegBlockInt32<8, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[2] = Add(lhs.buf.reg[2], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[3] = Add(lhs.buf.reg[3], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[4] = Add(lhs.buf.reg[4], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[5] = Add(lhs.buf.reg[5], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[6] = Add(lhs.buf.reg[6], DupLane<3>(rhs.buf.reg[0]));
    result.buf.reg[7] = Add(lhs.buf.reg[7], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 8x4 := 8x4 + 8x1
template <>
struct BroadcastAddImpl<RegBlockInt32<8, 4>, RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[1]);
    result.buf.reg[2] = Add(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] = Add(lhs.buf.reg[3], rhs.buf.reg[1]);
    result.buf.reg[4] = Add(lhs.buf.reg[4], rhs.buf.reg[0]);
    result.buf.reg[5] = Add(lhs.buf.reg[5], rhs.buf.reg[1]);
    result.buf.reg[6] = Add(lhs.buf.reg[6], rhs.buf.reg[0]);
    result.buf.reg[7] = Add(lhs.buf.reg[7], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x8
template <>
struct BroadcastAddImpl<RegBlockInt32<1, 8>, RegBlockInt32<1, 8>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 8>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] = Add(lhs.buf.reg[1], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x1
template <>
struct BroadcastAddImpl<RegBlockInt32<1, 8>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] = Add(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    result.buf.reg[1] = Add(lhs.buf.reg[1], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 1x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<4, 1>,
                                                      RegBlockInt32<1, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 1x4 := 1x4 + 1x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<1, 4>,
                                                      RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 + 4x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<4, 1>,
                                                      RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 + 1x4
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<1, 4>,
                                                      RegBlockInt32<1, 4>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 + 1x4
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<4, 4>,
                                                      RegBlockInt32<1, 4>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[1], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[2] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[2], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[3] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[3], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x4 := 4x4 + 4x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<4, 4>,
                                                      RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 4> result;
    result.buf.reg[0] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[1], rhs.buf.reg[0]);
    result.buf.reg[2] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[3], rhs.buf.reg[0]);
    return result;
  }
};

// 8x1 := 8x1 + 1x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<8, 1>,
                                                      RegBlockInt32<1, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    const Int32x4 p = Dup<Int32x4>(rhs.buf.reg[0]);
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = SaturatingRoundingDoublingHighMul(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 + 8x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<8, 1>,
                                                      RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] =
          SaturatingRoundingDoublingHighMul(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 + 1x4
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<8, 4>,
                                                      RegBlockInt32<1, 4>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[0], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[1] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[1], DupLane<0>(rhs.buf.reg[0]));
    result.buf.reg[2] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[2], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[3] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[3], DupLane<1>(rhs.buf.reg[0]));
    result.buf.reg[4] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[4], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[5] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[5], DupLane<2>(rhs.buf.reg[0]));
    result.buf.reg[6] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[6], DupLane<3>(rhs.buf.reg[0]));
    result.buf.reg[7] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[7], DupLane<3>(rhs.buf.reg[0]));
    return result;
  }
};

// 8x4 := 8x4 + 8x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<8, 4>,
                                                      RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 4> result;
    result.buf.reg[0] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[1], rhs.buf.reg[1]);
    result.buf.reg[2] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[2], rhs.buf.reg[0]);
    result.buf.reg[3] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[3], rhs.buf.reg[1]);
    result.buf.reg[4] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[4], rhs.buf.reg[0]);
    result.buf.reg[5] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[5], rhs.buf.reg[1]);
    result.buf.reg[6] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[6], rhs.buf.reg[0]);
    result.buf.reg[7] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[7], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x8
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<1, 8>,
                                                      RegBlockInt32<1, 8>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 8>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[0], rhs.buf.reg[0]);
    result.buf.reg[1] =
        SaturatingRoundingDoublingHighMul(lhs.buf.reg[1], rhs.buf.reg[1]);
    return result;
  }
};

// 1x8 := 1x8 + 1x1
template <>
struct BroadcastSaturatingRoundingDoublingHighMulImpl<RegBlockInt32<1, 8>,
                                                      RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 8> Run(const RegBlockInt32<1, 8>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 8> result;
    result.buf.reg[0] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    result.buf.reg[1] = SaturatingRoundingDoublingHighMul(
        lhs.buf.reg[1], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 * 1x1
template <>
struct BroadcastMulImpl<RegBlockInt32<4, 1>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = Mul(lhs.buf.reg[0], Dup<Int32x4>(rhs.buf.reg[0]));
    return result;
  }
};

// 4x1 := 4x1 * 4x1
template <>
struct BroadcastMulImpl<RegBlockInt32<4, 1>, RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 1> Run(const RegBlockInt32<4, 1>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 1> result;
    result.buf.reg[0] = Mul(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 * 1x4
template <>
struct BroadcastMulImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = Mul(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 1x4 := 1x4 * 1x1
template <>
struct BroadcastMulImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<1, 4> Run(const RegBlockInt32<1, 4>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<1, 4> result;
    result.buf.reg[0] = Mul(lhs.buf.reg[0], rhs.buf.reg[0]);
    return result;
  }
};

// 4x4 := 4x4 * 1x4
template <>
struct BroadcastMulImpl<RegBlockInt32<4, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<4, 4> result;
    const Int32x4 p = rhs.buf.reg[0];
    result.buf.reg[0] = MulByRhsLane<0>(lhs.buf.reg[0], p);
    result.buf.reg[1] = MulByRhsLane<1>(lhs.buf.reg[1], p);
    result.buf.reg[2] = MulByRhsLane<2>(lhs.buf.reg[2], p);
    result.buf.reg[3] = MulByRhsLane<3>(lhs.buf.reg[3], p);
    return result;
  }
};

// 4x4 := 4x4 * 4x1
template <>
struct BroadcastMulImpl<RegBlockInt32<4, 4>, RegBlockInt32<4, 1>> {
  static RegBlockInt32<4, 4> Run(const RegBlockInt32<4, 4>& lhs,
                                 const RegBlockInt32<4, 1>& rhs) {
    RegBlockInt32<4, 4> result;
    const Int32x4 p = rhs.buf.reg[0];
    result.buf.reg[0] = Mul(lhs.buf.reg[0], p);
    result.buf.reg[1] = Mul(lhs.buf.reg[1], p);
    result.buf.reg[2] = Mul(lhs.buf.reg[2], p);
    result.buf.reg[3] = Mul(lhs.buf.reg[3], p);
    return result;
  }
};

// 8x1 := 8x1 * 1x1
template <>
struct BroadcastMulImpl<RegBlockInt32<8, 1>, RegBlockInt32<1, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<1, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    const std::int32_t p = rhs.buf.reg[0];
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Mul(lhs.buf.reg[i], p);
    }
    return result;
  }
};

// 8x1 := 8x1 * 8x1
template <>
struct BroadcastMulImpl<RegBlockInt32<8, 1>, RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 1> Run(const RegBlockInt32<8, 1>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 1> result;
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i] = Mul(lhs.buf.reg[i], rhs.buf.reg[i]);
    }
    return result;
  }
};

// 8x4 := 8x4 * 1x4
template <>
struct BroadcastMulImpl<RegBlockInt32<8, 4>, RegBlockInt32<1, 4>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<1, 4>& rhs) {
    RegBlockInt32<8, 4> result;
    const Int32x4 p = rhs.buf.reg[0];
    for (int i = 0; i < 2; i++) {
      result.buf.reg[i + 0] = MulByRhsLane<0>(lhs.buf.reg[i + 0], p);
      result.buf.reg[i + 2] = MulByRhsLane<1>(lhs.buf.reg[i + 2], p);
      result.buf.reg[i + 4] = MulByRhsLane<2>(lhs.buf.reg[i + 4], p);
      result.buf.reg[i + 6] = MulByRhsLane<3>(lhs.buf.reg[i + 6], p);
    }
    return result;
  }
};

// 8x4 := 8x4 * 8x1
template <>
struct BroadcastMulImpl<RegBlockInt32<8, 4>, RegBlockInt32<8, 1>> {
  static RegBlockInt32<8, 4> Run(const RegBlockInt32<8, 4>& lhs,
                                 const RegBlockInt32<8, 1>& rhs) {
    RegBlockInt32<8, 4> result;
    const Int32x4 p[2]{rhs.buf.reg[0], rhs.buf.reg[1]};
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 2; j++) {
        const int k = j + 2 * i;
        result.buf.reg[k] = Mul(lhs.buf.reg[k], p[j]);
      }
    }
    return result;
  }
};

// Rx1 += Rx1 * 1x1
template <int Rows>
struct BroadcastMulAddImpl<RegBlockInt32<Rows, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<Rows, 1>> {
  static void Run(const RegBlockInt32<Rows, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<Rows, 1>* acc) {
    const std::int32_t p = rhs.buf.reg[0];
    for (int i = 0; i < RegBlockInt32<Rows, 1>::kRegisterCount; i++) {
      MulAdd(lhs.buf.reg[i], p, &acc->buf.reg[i]);
    }
  }
};

// RxC += Rx1 * 1x1
template <int Rows, int Cols>
struct BroadcastMulAddImpl<RegBlockInt32<Rows, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<Rows, Cols>> {
  static void Run(const RegBlockInt32<Rows, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs,
                  RegBlockInt32<Rows, Cols>* acc) {
    const std::int32_t p = rhs.buf.reg[0];
    static constexpr int kRegsPerCol = RegBlockInt32<Rows, 1>::kRegisterCount;
    for (int i = 0; i < kRegsPerCol; i++) {
      const Int32x4 q = Mul(lhs.buf.reg[i], p);
      for (int j = 0; j < Cols; j++) {
        acc->buf.reg[i + j * kRegsPerCol] =
            Add(acc->buf.reg[i + j * kRegsPerCol], q);
      }
    }
  }
};

// 1xC += 1xC * 1x1
template <int Cols>
struct BroadcastMulAddImpl<RegBlockInt32<1, Cols>, RegBlockInt32<1, 1>,
                           RegBlockInt32<1, Cols>> {
  static void Run(const RegBlockInt32<1, Cols>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<1, Cols>* acc) {
    const std::int32_t p = rhs.buf.reg[0];
    for (int i = 0; i < RegBlockInt32<1, Cols>::kRegisterCount; i++) {
      MulAdd(lhs.buf.reg[i], p, &acc->buf.reg[i]);
    }
  }
};

// RxC += 1x1 * 1x1
template <int Rows, int Cols>
struct BroadcastMulAddImpl<RegBlockInt32<1, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<Rows, Cols>> {
  static void Run(const RegBlockInt32<1, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs,
                  RegBlockInt32<Rows, Cols>* acc) {
    const Int32x4 p = Dup<Int32x4>(Mul(lhs.buf.reg[0], rhs.buf.reg[0]));
    for (int i = 0; i < RegBlockInt32<Rows, Cols>::kRegisterCount; i++) {
      acc->buf.reg[i] = Add(acc->buf.reg[i], p);
    }
  }
};

// 1x1 += 1x1 * 1x1
template <>
struct BroadcastMulAddImpl<RegBlockInt32<1, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<1, 1>> {
  static void Run(const RegBlockInt32<1, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<1, 1>* acc) {
    MulAdd(lhs.buf.reg[0], rhs.buf.reg[0], &acc->buf.reg[0]);
  }
};

// Rx4 += Rx1 * 1x4
template <int Rows>
struct BroadcastMulAddImpl<RegBlockInt32<Rows, 1>, RegBlockInt32<1, 4>,
                           RegBlockInt32<Rows, 4>> {
  static void Run(const RegBlockInt32<Rows, 1>& lhs,
                  const RegBlockInt32<1, 4>& rhs, RegBlockInt32<Rows, 4>* acc) {
    const Int32x4 p = rhs.buf.reg[0];
    static constexpr int kRegsPerCol = RegBlockInt32<Rows, 1>::kRegisterCount;
    for (int i = 0; i < kRegsPerCol; i++) {
      MulAddByRhsLane<0>(lhs.buf.reg[i], p, &acc->buf.reg[i + 0 * kRegsPerCol]);
      MulAddByRhsLane<1>(lhs.buf.reg[i], p, &acc->buf.reg[i + 1 * kRegsPerCol]);
      MulAddByRhsLane<2>(lhs.buf.reg[i], p, &acc->buf.reg[i + 2 * kRegsPerCol]);
      MulAddByRhsLane<3>(lhs.buf.reg[i], p, &acc->buf.reg[i + 3 * kRegsPerCol]);
    }
  }
};

// Rx4 += 1x4 * 1x1
template <int Rows>
struct BroadcastMulAddImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 1>,
                           RegBlockInt32<Rows, 4>> {
  static void Run(const RegBlockInt32<1, 4>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<Rows, 4>* acc) {
    const Int32x4 p = Mul(lhs.buf.reg[0], rhs.buf.reg[0]);
    Int32x4 q[4];
    q[0] = DupLane<0>(p);
    q[1] = DupLane<1>(p);
    q[2] = DupLane<2>(p);
    q[3] = DupLane<3>(p);
    static constexpr int kRegsPerCol = RegBlockInt32<Rows, 1>::kRegisterCount;
    for (int i = 0; i < kRegsPerCol; i++) {
      for (int j = 0; j < 4; j++) {
        acc->buf.reg[i + j * kRegsPerCol] =
            Add(q[j], acc->buf.reg[i + j * kRegsPerCol]);
      }
    }
  }
};

// 1xC += 1x1 * 1x1
template <int Cols>
struct BroadcastMulAddImpl<RegBlockInt32<1, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<1, Cols>> {
  static void Run(const RegBlockInt32<1, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<1, Cols>* acc) {
    const Int32x4 p = Dup<Int32x4>(Mul(lhs.buf.reg[0], rhs.buf.reg[0]));
    for (int i = 0; i < RegBlockInt32<1, Cols>::kRegisterCount; i++) {
      acc->buf.reg[i] = Add(acc->buf.reg[i], p);
    }
  }
};

// 1x4 += 1x4 * 1x1
template <>
struct BroadcastMulAddImpl<RegBlockInt32<1, 4>, RegBlockInt32<1, 1>,
                           RegBlockInt32<1, 4>> {
  static void Run(const RegBlockInt32<1, 4>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<1, 4>* acc) {
    const std::int32_t p = rhs.buf.reg[0];
    MulAdd(lhs.buf.reg[0], p, &acc->buf.reg[0]);
  }
};

// 4xC += 4x1 * 1x1
template <int Cols>
struct BroadcastMulAddImpl<RegBlockInt32<4, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<4, Cols>> {
  static void Run(const RegBlockInt32<4, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<4, Cols>* acc) {
    const Int32x4 p = Mul(lhs.buf.reg[0], rhs.buf.reg[0]);
    for (int i = 0; i < Cols; i++) {
      acc->buf.reg[i] = Add(p, acc->buf.reg[i]);
    }
  }
};

// 4x1 += 4x1 * 1x1
template <>
struct BroadcastMulAddImpl<RegBlockInt32<4, 1>, RegBlockInt32<1, 1>,
                           RegBlockInt32<4, 1>> {
  static void Run(const RegBlockInt32<4, 1>& lhs,
                  const RegBlockInt32<1, 1>& rhs, RegBlockInt32<4, 1>* acc) {
    const std::int32_t p = rhs.buf.reg[0];
    MulAdd(lhs.buf.reg[0], p, &acc->buf.reg[0]);
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_SIMD_WRAPPERS_COMMON_NEON_SSE_H_
