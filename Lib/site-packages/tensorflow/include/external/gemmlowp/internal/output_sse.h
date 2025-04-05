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

// output_sse.h: optimized SSE4.2 specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_SSE_H_
#define GEMMLOWP_INTERNAL_OUTPUT_SSE_H_

#include "output.h"

#include <smmintrin.h>

namespace gemmlowp {

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToUint8,
                                 RegBufferInt32<4>> {
  typedef RegBufferInt32<4> InputType;
  typedef RegBufferUint8<4> OutputType;

  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    __m128i res_16 = _mm_packs_epi32(input.reg[0], input.reg[0]);
    __m128i res_8 = _mm_packus_epi16(res_16, res_16);
    output.reg[0] = _mm_cvtsi128_si32(res_8);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToUint8,
                                 RegBufferInt32<8>> {
  typedef RegBufferInt32<8> InputType;
  typedef RegBufferUint8<8> OutputType;

  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    __m128i res_16 = _mm_packs_epi32(input.reg[0], input.reg[1]);
    __m128i res_8 = _mm_packus_epi16(res_16, res_16);
    output.reg[0] = _mm_extract_epi32(res_8, 0);
    output.reg[1] = _mm_extract_epi32(res_8, 1);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToUint8,
                                 RegBufferInt32<16>> {
  typedef RegBufferInt32<16> InputType;
  typedef RegBufferUint8<16> OutputType;

  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    __m128i res_16_0 = _mm_packs_epi32(input.reg[0], input.reg[1]);
    __m128i res_16_1 = _mm_packs_epi32(input.reg[2], input.reg[3]);
    output.reg[0] = _mm_packus_epi16(res_16_0, res_16_1);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToUint8,
                                 RegBufferInt32<32>> {
  typedef RegBufferInt32<32> InputType;
  typedef RegBufferUint8<32> OutputType;

  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    __m128i res_16_0 = _mm_packs_epi32(input.reg[0], input.reg[1]);
    __m128i res_16_1 = _mm_packs_epi32(input.reg[2], input.reg[3]);
    output.reg[0] = _mm_packus_epi16(res_16_0, res_16_1);
    __m128i res_16_2 = _mm_packs_epi32(input.reg[4], input.reg[5]);
    __m128i res_16_3 = _mm_packs_epi32(input.reg[6], input.reg[7]);
    output.reg[1] = _mm_packus_epi16(res_16_2, res_16_3);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegBufferInt32<4>> {
  typedef RegBufferInt32<4> InputType;
  typedef RegBufferInt16<4> OutputType;

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    __m128i res_16 = _mm_packs_epi32(input.reg[0], input.reg[0]);
    output.reg[0] = _mm_extract_epi16(res_16, 0);
    output.reg[1] = _mm_extract_epi16(res_16, 1);
    output.reg[2] = _mm_extract_epi16(res_16, 2);
    output.reg[3] = _mm_extract_epi16(res_16, 3);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegBufferInt32<8>> {
  typedef RegBufferInt32<8> InputType;
  typedef RegBufferInt16<8> OutputType;

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    output.reg[0] = _mm_packs_epi32(input.reg[0], input.reg[1]);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegBufferInt32<16>> {
  typedef RegBufferInt32<16> InputType;
  typedef RegBufferInt16<16> OutputType;

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    output.reg[0] = _mm_packs_epi32(input.reg[0], input.reg[1]);
    output.reg[1] = _mm_packs_epi32(input.reg[2], input.reg[3]);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegBufferInt32<32>> {
  typedef RegBufferInt32<32> InputType;
  typedef RegBufferInt16<32> OutputType;

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    output.reg[0] = _mm_packs_epi32(input.reg[0], input.reg[1]);
    output.reg[1] = _mm_packs_epi32(input.reg[2], input.reg[3]);
    output.reg[2] = _mm_packs_epi32(input.reg[4], input.reg[5]);
    output.reg[3] = _mm_packs_epi32(input.reg[6], input.reg[7]);
    return output;
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<4, 1>, DstType> {
  static void Run(const RegBlockInt32<4, 1>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      StoreInt32x4(dst->data(row, col), src.buf.reg[0]);
    } else {
      *dst->data(row + 0, col) = GetLane<0>(src.buf.reg[0]);
      *dst->data(row + 1, col) = GetLane<1>(src.buf.reg[0]);
      *dst->data(row + 2, col) = GetLane<2>(src.buf.reg[0]);
      *dst->data(row + 3, col) = GetLane<3>(src.buf.reg[0]);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<8, 1>, DstType> {
  static void Run(const RegBlockInt32<8, 1>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      StoreInt32x4(dst->data(row, col), src.buf.reg[0]);
      StoreInt32x4(dst->data(row + 4, col), src.buf.reg[1]);
    } else {
      *dst->data(row + 0, col) = GetLane<0>(src.buf.reg[0]);
      *dst->data(row + 1, col) = GetLane<1>(src.buf.reg[0]);
      *dst->data(row + 2, col) = GetLane<2>(src.buf.reg[0]);
      *dst->data(row + 3, col) = GetLane<3>(src.buf.reg[0]);
      *dst->data(row + 4, col) = GetLane<0>(src.buf.reg[1]);
      *dst->data(row + 5, col) = GetLane<1>(src.buf.reg[1]);
      *dst->data(row + 6, col) = GetLane<2>(src.buf.reg[1]);
      *dst->data(row + 7, col) = GetLane<3>(src.buf.reg[1]);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<4, 1>, DstType> {
  static void Run(const RegBlockInt16<4, 1>& src, DstType* dst, int row,
                  int col) {
    *dst->data(row + 0, col) = src.buf.reg[0];
    *dst->data(row + 1, col) = src.buf.reg[1];
    *dst->data(row + 2, col) = src.buf.reg[2];
    *dst->data(row + 3, col) = src.buf.reg[3];
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<8, 1>, DstType> {
  static void Run(const RegBlockInt16<8, 1>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      StoreInt16x8(dst->data(row, col), src.buf.reg[0]);
    } else {
      *dst->data(row + 0, col) = _mm_extract_epi16(src.buf.reg[0], 0);
      *dst->data(row + 1, col) = _mm_extract_epi16(src.buf.reg[0], 1);
      *dst->data(row + 2, col) = _mm_extract_epi16(src.buf.reg[0], 2);
      *dst->data(row + 3, col) = _mm_extract_epi16(src.buf.reg[0], 3);
      *dst->data(row + 4, col) = _mm_extract_epi16(src.buf.reg[0], 4);
      *dst->data(row + 5, col) = _mm_extract_epi16(src.buf.reg[0], 5);
      *dst->data(row + 6, col) = _mm_extract_epi16(src.buf.reg[0], 6);
      *dst->data(row + 7, col) = _mm_extract_epi16(src.buf.reg[0], 7);
    }
  }
};

inline RegBlockInt32<4, 4> Transpose(const RegBlockInt32<4, 4>& src) {
  __m128i t0 = _mm_unpacklo_epi32(src.buf.reg[0], src.buf.reg[1]);
  __m128i t1 = _mm_unpacklo_epi32(src.buf.reg[2], src.buf.reg[3]);
  __m128i t2 = _mm_unpackhi_epi32(src.buf.reg[0], src.buf.reg[1]);
  __m128i t3 = _mm_unpackhi_epi32(src.buf.reg[2], src.buf.reg[3]);

  RegBlockInt32<4, 4> result;
  result.buf.reg[0] = _mm_unpacklo_epi64(t0, t1);
  result.buf.reg[1] = _mm_unpackhi_epi64(t0, t1);
  result.buf.reg[2] = _mm_unpacklo_epi64(t2, t3);
  result.buf.reg[3] = _mm_unpackhi_epi64(t2, t3);
  return result;
}

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<4, 4>, DstType> {
  static void Run(const RegBlockInt32<4, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row, col + i), src.buf.reg[i]);
      }
    } else {
      const auto transpose = Transpose(src);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + i, col), transpose.buf.reg[i]);
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<4, 4>, DstType> {
  static void Run(const RegBlockInt16<4, 4>& src, DstType* dst, int row,
                  int col) {
    std::int16_t buf[16];
    StoreInt16x8(buf + 0, src.buf.reg[0]);
    StoreInt16x8(buf + 8, src.buf.reg[1]);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        *dst->data(row + i, col + j) = buf[i + 4 * j];
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<8, 4>, DstType> {
  static void Run(const RegBlockInt32<8, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row, col + i), src.buf.reg[2 * i]);
        StoreInt32x4(dst->data(row + 4, col + i), src.buf.reg[2 * i + 1]);
      }
    } else {
      RegBlockInt32<4, 4> top;
      top.buf.reg[0] = src.buf.reg[0];
      top.buf.reg[1] = src.buf.reg[2];
      top.buf.reg[2] = src.buf.reg[4];
      top.buf.reg[3] = src.buf.reg[6];
      const auto transpose_top = Transpose(top);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + i, col), transpose_top.buf.reg[i]);
      }
      RegBlockInt32<4, 4> bottom;
      bottom.buf.reg[0] = src.buf.reg[1];
      bottom.buf.reg[1] = src.buf.reg[3];
      bottom.buf.reg[2] = src.buf.reg[5];
      bottom.buf.reg[3] = src.buf.reg[7];
      const auto transpose_bottom = Transpose(bottom);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + 4 + i, col), transpose_bottom.buf.reg[i]);
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<8, 4>, DstType> {
  static void Run(const RegBlockInt16<8, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      for (int i = 0; i < 4; i++) {
        StoreInt16x8(dst->data(row, col + i), src.buf.reg[i]);
      }
    } else {
      std::int16_t buf[32];
      StoreInt16x8(buf + 0, src.buf.reg[0]);
      StoreInt16x8(buf + 8, src.buf.reg[1]);
      StoreInt16x8(buf + 16, src.buf.reg[2]);
      StoreInt16x8(buf + 24, src.buf.reg[3]);
      for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
          *dst->data(row + i, col + j) = buf[i + 8 * j];
        }
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<8, 8>, DstType> {
  static void Run(const RegBlockInt32<8, 8>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      for (int i = 0; i < 8; i++) {
        StoreInt32x4(dst->data(row, col + i), src.buf.reg[2 * i]);
        StoreInt32x4(dst->data(row + 4, col + i), src.buf.reg[2 * i + 1]);
      }
    } else {
      RegBlockInt32<4, 4> top_left;
      top_left.buf.reg[0] = src.buf.reg[0];
      top_left.buf.reg[1] = src.buf.reg[2];
      top_left.buf.reg[2] = src.buf.reg[4];
      top_left.buf.reg[3] = src.buf.reg[6];
      const auto transpose_top_left = Transpose(top_left);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + i, col), transpose_top_left.buf.reg[i]);
      }
      RegBlockInt32<4, 4> bottom_left;
      bottom_left.buf.reg[0] = src.buf.reg[1];
      bottom_left.buf.reg[1] = src.buf.reg[3];
      bottom_left.buf.reg[2] = src.buf.reg[5];
      bottom_left.buf.reg[3] = src.buf.reg[7];
      const auto transpose_bottom_left = Transpose(bottom_left);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + 4 + i, col),
                     transpose_bottom_left.buf.reg[i]);
      }
      RegBlockInt32<4, 4> top_right;
      top_right.buf.reg[0] = src.buf.reg[8];
      top_right.buf.reg[1] = src.buf.reg[10];
      top_right.buf.reg[2] = src.buf.reg[12];
      top_right.buf.reg[3] = src.buf.reg[14];
      const auto transpose_top_right = Transpose(top_right);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + i, col + 4),
                     transpose_top_right.buf.reg[i]);
      }
      RegBlockInt32<4, 4> bottom_right;
      bottom_right.buf.reg[0] = src.buf.reg[9];
      bottom_right.buf.reg[1] = src.buf.reg[11];
      bottom_right.buf.reg[2] = src.buf.reg[13];
      bottom_right.buf.reg[3] = src.buf.reg[15];
      const auto transpose_bottom_right = Transpose(bottom_right);
      for (int i = 0; i < 4; i++) {
        StoreInt32x4(dst->data(row + 4 + i, col + 4),
                     transpose_bottom_right.buf.reg[i]);
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<8, 8>, DstType> {
  static void Run(const RegBlockInt16<8, 8>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      for (int i = 0; i < 8; i++) {
        StoreInt16x8(dst->data(row, col + i), src.buf.reg[i]);
      }
    } else {
      // top-left 4x4
      __m128i t0 = _mm_unpacklo_epi16(src.buf.reg[0], src.buf.reg[1]);
      __m128i t1 = _mm_unpacklo_epi16(src.buf.reg[2], src.buf.reg[3]);
      __m128i u0 = _mm_unpacklo_epi32(t0, t1);
      __m128i u1 = _mm_unpackhi_epi32(t0, t1);
      // top-right 4x4
      __m128i t2 = _mm_unpacklo_epi16(src.buf.reg[4], src.buf.reg[5]);
      __m128i t3 = _mm_unpacklo_epi16(src.buf.reg[6], src.buf.reg[7]);
      __m128i u2 = _mm_unpacklo_epi32(t2, t3);
      __m128i u3 = _mm_unpackhi_epi32(t2, t3);
      // bottom-left 4x4
      __m128i t4 = _mm_unpackhi_epi16(src.buf.reg[0], src.buf.reg[1]);
      __m128i t5 = _mm_unpackhi_epi16(src.buf.reg[2], src.buf.reg[3]);
      __m128i u4 = _mm_unpacklo_epi32(t4, t5);
      __m128i u5 = _mm_unpackhi_epi32(t4, t5);
      // bottom-right 4x4
      __m128i t6 = _mm_unpackhi_epi16(src.buf.reg[4], src.buf.reg[5]);
      __m128i t7 = _mm_unpackhi_epi16(src.buf.reg[6], src.buf.reg[7]);
      __m128i u6 = _mm_unpacklo_epi32(t6, t7);
      __m128i u7 = _mm_unpackhi_epi32(t6, t7);

      StoreInt16x8(dst->data(row + 0, col), _mm_unpacklo_epi64(u0, u2));
      StoreInt16x8(dst->data(row + 1, col), _mm_unpackhi_epi64(u0, u2));
      StoreInt16x8(dst->data(row + 2, col), _mm_unpacklo_epi64(u1, u3));
      StoreInt16x8(dst->data(row + 3, col), _mm_unpackhi_epi64(u1, u3));
      StoreInt16x8(dst->data(row + 4, col), _mm_unpacklo_epi64(u4, u6));
      StoreInt16x8(dst->data(row + 5, col), _mm_unpackhi_epi64(u4, u6));
      StoreInt16x8(dst->data(row + 6, col), _mm_unpacklo_epi64(u5, u7));
      StoreInt16x8(dst->data(row + 7, col), _mm_unpackhi_epi64(u5, u7));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<1, 4>, DstType> {
  static void Run(const RegBlockInt32<1, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      *dst->data(row, col + 0) = GetLane<0>(src.buf.reg[0]);
      *dst->data(row, col + 1) = GetLane<1>(src.buf.reg[0]);
      *dst->data(row, col + 2) = GetLane<2>(src.buf.reg[0]);
      *dst->data(row, col + 3) = GetLane<3>(src.buf.reg[0]);
    } else {
      StoreInt32x4(dst->data(row, col), src.buf.reg[0]);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<4, 1>, DstType> {
  static void Run(const RegBlockUint8<4, 1>& src, DstType* dst, int row,
                  int col) {
    const std::uint32_t src_reg = src.buf.reg[0];
    for (int i = 0; i < 4; i++) {
      *dst->data(row + i, col) = (src_reg >> (8 * i));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 1>, DstType> {
  static void Run(const RegBlockUint8<8, 1>& src, DstType* dst, int row,
                  int col) {
    for (int i = 0; i < 4; i++) {
      *dst->data(row + i, col) = (src.buf.reg[0] >> (8 * i));
    }
    for (int i = 0; i < 4; i++) {
      *dst->data(row + 4 + i, col) = (src.buf.reg[1] >> (8 * i));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<1, 4>, DstType> {
  static void Run(const RegBlockUint8<1, 4>& src, DstType* dst, int row,
                  int col) {
    for (int i = 0; i < 4; i++) {
      *dst->data(row, col + i) = (src.buf.reg[0] >> (8 * i));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<4, 4>, DstType> {
  static void Run(const RegBlockUint8<4, 4>& src, DstType* dst, int row,
                  int col) {
    std::uint8_t buf[16];
    StoreUint8x16(buf, src.buf.reg[0]);
    for (int c = 0; c < 4; c++) {
      for (int r = 0; r < 4; r++) {
        *dst->data(row + r, col + c) = buf[r + 4 * c];
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 4>, DstType> {
  static void Run(const RegBlockUint8<8, 4>& src, DstType* dst, int row,
                  int col) {
    std::uint8_t buf[32];
    StoreUint8x16(buf, src.buf.reg[0]);
    StoreUint8x16(buf + 16, src.buf.reg[1]);
    for (int c = 0; c < 4; c++) {
      for (int r = 0; r < 8; r++) {
        *dst->data(row + r, col + c) = buf[r + 8 * c];
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 8>, DstType> {
  static void Run(const RegBlockUint8<8, 8>& src, DstType* dst, int row,
                  int col) {
    std::uint8_t buf[64];
    StoreUint8x16(buf, src.buf.reg[0]);
    StoreUint8x16(buf + 16, src.buf.reg[1]);
    StoreUint8x16(buf + 32, src.buf.reg[2]);
    StoreUint8x16(buf + 48, src.buf.reg[3]);
    for (int c = 0; c < 8; c++) {
      for (int r = 0; r < 8; r++) {
        *dst->data(row + r, col + c) = buf[r + 8 * c];
      }
    }
  }
};

// Specialization for MatrixMap, for performance.
template <typename tScalar, MapOrder tOrder>
struct StoreFinalOutputImpl<RegBlockUint8<8, 8>, MatrixMap<tScalar, tOrder>> {
  static void Run(const RegBlockUint8<8, 8>& src,
                  MatrixMap<tScalar, tOrder>* dst, int row, int col) {
    std::uint8_t buf[64];
    StoreUint8x16(buf, src.buf.reg[0]);
    StoreUint8x16(buf + 16, src.buf.reg[1]);
    StoreUint8x16(buf + 32, src.buf.reg[2]);
    StoreUint8x16(buf + 48, src.buf.reg[3]);
    // Make a local copy so that the compiler can prove that data_ does not
    // alias &data_ or &stride_.
    MatrixMap<tScalar, tOrder> local = *dst;
    for (int c = 0; c < 8; c++) {
      for (int r = 0; r < 8; r++) {
        *local.data(row + r, col + c) = buf[r + 8 * c];
      }
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_SSE_H_
