// Copyright 2018 The Gemmlowp Authors. All Rights Reserved.
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

// output_msa.h: optimized MSA specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_MSA_H_
#define GEMMLOWP_INTERNAL_OUTPUT_MSA_H_

#include "output.h"

#include <msa.h>

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
    // Signed saturate each 32-bit element to 9 bits
    // (this takes full care of non-negative elements).
    v4i32 tmp = __builtin_msa_sat_s_w(input.reg[0], 8);
    // Zero out negative elements.
    tmp = __builtin_msa_maxi_s_w(tmp, 0);
    // Pack every 32-bit element into 16 bits.
    tmp = reinterpret_cast<v4i32>(__builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(tmp), reinterpret_cast<v8i16>(tmp)));
    // Pack every element into 8 bits.
    tmp = reinterpret_cast<v4i32>(__builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp), reinterpret_cast<v16i8>(tmp)));
    // Return 4 uint8_t elements as uint32_t.
    output.reg[0] = __builtin_msa_copy_s_w(tmp, 0);
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
    // Signed saturate each 32-bit element to 9 bits
    // (this takes full care of non-negative elements).
    v4i32 tmp_lo = __builtin_msa_sat_s_w(input.reg[0], 8);
    v4i32 tmp_hi = __builtin_msa_sat_s_w(input.reg[1], 8);
    // Pack every 32-bit element into 16 bits,
    // combining all 8 elements into one vector.
    tmp_lo = reinterpret_cast<v4i32>(__builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(tmp_hi), reinterpret_cast<v8i16>(tmp_lo)));
    // Zero out negative elements.
    tmp_lo = reinterpret_cast<v4i32>(__builtin_msa_maxi_s_h(
        reinterpret_cast<v8i16>(tmp_lo), 0));
    // Pack every element into 8 bits.
    tmp_lo = reinterpret_cast<v4i32>(__builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp_lo), reinterpret_cast<v16i8>(tmp_lo)));
    // Return 8 uint8_t elements as 2 uint32_t's.
    output.reg[0] = __builtin_msa_copy_s_w(tmp_lo, 0);
    output.reg[1] = __builtin_msa_copy_s_w(tmp_lo, 1);
    return output;
  }
};

#define GEMMLOWP_MIPS_SAT_U8_16(out, in0, in1, in2, in3)                     \
  {                                                                          \
    v4i32 tmp0 = __builtin_msa_sat_s_w(in0, 8);                              \
    v4i32 tmp1 = __builtin_msa_sat_s_w(in1, 8);                              \
    v4i32 tmp2 = __builtin_msa_sat_s_w(in2, 8);                              \
    v4i32 tmp3 = __builtin_msa_sat_s_w(in3, 8);                              \
    tmp0 = reinterpret_cast<v4i32>(__builtin_msa_pckev_h(                    \
        reinterpret_cast<v8i16>(tmp1), reinterpret_cast<v8i16>(tmp0)));      \
    tmp2 = reinterpret_cast<v4i32>(__builtin_msa_pckev_h(                    \
        reinterpret_cast<v8i16>(tmp3), reinterpret_cast<v8i16>(tmp2)));      \
    tmp0 = reinterpret_cast<v4i32>(__builtin_msa_maxi_s_h(                   \
        reinterpret_cast<v8i16>(tmp0), 0));                                  \
    tmp2 = reinterpret_cast<v4i32>(__builtin_msa_maxi_s_h(                   \
        reinterpret_cast<v8i16>(tmp2), 0));                                  \
    tmp0 = reinterpret_cast<v4i32>(__builtin_msa_pckev_b(                    \
        reinterpret_cast<v16i8>(tmp2), reinterpret_cast<v16i8>(tmp0)));      \
    out = reinterpret_cast<v16i8>(tmp0);                                     \
  }

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToUint8,
                                 RegBufferInt32<16>> {
  typedef RegBufferInt32<16> InputType;
  typedef RegBufferUint8<16> OutputType;

  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    GEMMLOWP_MIPS_SAT_U8_16(output.reg[0], input.reg[0], input.reg[1],
                            input.reg[2], input.reg[3]);
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
    GEMMLOWP_MIPS_SAT_U8_16(output.reg[0], input.reg[0], input.reg[1],
                            input.reg[2], input.reg[3]);
    GEMMLOWP_MIPS_SAT_U8_16(output.reg[1], input.reg[4], input.reg[5],
                            input.reg[6], input.reg[7]);
    return output;
  }
};

#undef GEMMLOWP_MIPS_SAT_U8_16

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegBufferInt32<4>> {
  typedef RegBufferInt32<4> InputType;
  typedef RegBufferInt16<4> OutputType;

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    // Signed saturate each 32-bit element to 16 bits.
    v8i16 tmp =
        reinterpret_cast<v8i16>(__builtin_msa_sat_s_w(input.reg[0], 15));
    output.reg[0] = __builtin_msa_copy_s_h(tmp, 0);
    output.reg[1] = __builtin_msa_copy_s_h(tmp, 2);
    output.reg[2] = __builtin_msa_copy_s_h(tmp, 4);
    output.reg[3] = __builtin_msa_copy_s_h(tmp, 6);
    return output;
  }
};

#define GEMMLOWP_MIPS_SAT_I16_8(out, in0, in1)                  \
  {                                                             \
    v4i32 tmp0 = __builtin_msa_sat_s_w(in0, 15);                \
    v4i32 tmp1 = __builtin_msa_sat_s_w(in1, 15);                \
    out = __builtin_msa_pckev_h(reinterpret_cast<v8i16>(tmp1),  \
                                reinterpret_cast<v8i16>(tmp0)); \
  }

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegBufferInt32<8>> {
  typedef RegBufferInt32<8> InputType;
  typedef RegBufferInt16<8> OutputType;

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[0], input.reg[0], input.reg[1]);
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
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[0], input.reg[0], input.reg[1]);
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[1], input.reg[2], input.reg[3]);
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
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[0], input.reg[0], input.reg[1]);
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[1], input.reg[2], input.reg[3]);
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[2], input.reg[4], input.reg[5]);
    GEMMLOWP_MIPS_SAT_I16_8(output.reg[3], input.reg[6], input.reg[7]);
    return output;
  }
};

#undef GEMMLOWP_MIPS_SAT_I16_8

template <>
struct OutputStageEvalBufferImpl<OutputStageTruncatingCastToUint8,
                                 RegBufferInt32<4>> {
  typedef RegBufferInt32<4> InputType;
  typedef RegBufferUint8<4> OutputType;

  typedef OutputStageTruncatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    // Pack every 32-bit element into 16 bits.
    v4i32 tmp = reinterpret_cast<v4i32>(__builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[0]),
        reinterpret_cast<v8i16>(input.reg[0])));
    // Pack every element into 8 bits.
    tmp = reinterpret_cast<v4i32>(__builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp), reinterpret_cast<v16i8>(tmp)));
    // Return 4 uint8_t elements as uint32_t.
    output.reg[0] = __builtin_msa_copy_s_w(tmp, 0);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageTruncatingCastToUint8,
                                 RegBufferInt32<8>> {
  typedef RegBufferInt32<8> InputType;
  typedef RegBufferUint8<8> OutputType;

  typedef OutputStageTruncatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    // Pack every 32-bit element into 16 bits.
    v4i32 tmp = reinterpret_cast<v4i32>(__builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[1]),
        reinterpret_cast<v8i16>(input.reg[0])));
    // Pack every element into 8 bits.
    tmp = reinterpret_cast<v4i32>(__builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp), reinterpret_cast<v16i8>(tmp)));
    // Return 8 uint8_t elements as 2 uint32_t's.
    output.reg[0] = __builtin_msa_copy_s_w(tmp, 0);
    output.reg[1] = __builtin_msa_copy_s_w(tmp, 1);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageTruncatingCastToUint8,
                                 RegBufferInt32<16>> {
  typedef RegBufferInt32<16> InputType;
  typedef RegBufferUint8<16> OutputType;

  typedef OutputStageTruncatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    // Pack every 32-bit element into 16 bits.
    v8i16 tmp0 = __builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[1]),
        reinterpret_cast<v8i16>(input.reg[0]));
    v8i16 tmp1 = __builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[3]),
        reinterpret_cast<v8i16>(input.reg[2]));
    // Pack every element into 8 bits.
    output.reg[0] = __builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp1), reinterpret_cast<v16i8>(tmp0));
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageTruncatingCastToUint8,
                                 RegBufferInt32<32>> {
  typedef RegBufferInt32<32> InputType;
  typedef RegBufferUint8<32> OutputType;

  typedef OutputStageTruncatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    // Pack every 32-bit element into 16 bits.
    v8i16 tmp0 = __builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[1]),
        reinterpret_cast<v8i16>(input.reg[0]));
    v8i16 tmp1 = __builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[3]),
        reinterpret_cast<v8i16>(input.reg[2]));
    v8i16 tmp2 = __builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[5]),
        reinterpret_cast<v8i16>(input.reg[4]));
    v8i16 tmp3 = __builtin_msa_pckev_h(
        reinterpret_cast<v8i16>(input.reg[7]),
        reinterpret_cast<v8i16>(input.reg[6]));
    // Pack every element into 8 bits.
    output.reg[0] = __builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp1), reinterpret_cast<v16i8>(tmp0));
    output.reg[1] = __builtin_msa_pckev_b(
        reinterpret_cast<v16i8>(tmp3), reinterpret_cast<v16i8>(tmp2));
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
      *dst->data(row + 0, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 0);
      *dst->data(row + 1, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 1);
      *dst->data(row + 2, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 2);
      *dst->data(row + 3, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 3);
      *dst->data(row + 4, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 4);
      *dst->data(row + 5, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 5);
      *dst->data(row + 6, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 6);
      *dst->data(row + 7, col) = __builtin_msa_copy_s_h(src.buf.reg[0], 7);
    }
  }
};

inline RegBlockInt32<4, 4> Transpose(const RegBlockInt32<4, 4>& src) {
  RegBlockInt32<4, 4> result;
  v4i32 tmp0, tmp1;
  tmp0 = __builtin_msa_ilvr_w(src.buf.reg[1], src.buf.reg[0]);
  tmp1 = __builtin_msa_ilvr_w(src.buf.reg[3], src.buf.reg[2]);
  result.buf.reg[0] = reinterpret_cast<v4i32>(__builtin_msa_ilvr_d(
      reinterpret_cast<v2i64>(tmp1), reinterpret_cast<v2i64>(tmp0)));
  result.buf.reg[1] = reinterpret_cast<v4i32>(__builtin_msa_ilvl_d(
      reinterpret_cast<v2i64>(tmp1), reinterpret_cast<v2i64>(tmp0)));
  tmp0 = __builtin_msa_ilvl_w(src.buf.reg[1], src.buf.reg[0]);
  tmp1 = __builtin_msa_ilvl_w(src.buf.reg[3], src.buf.reg[2]);
  result.buf.reg[2] = reinterpret_cast<v4i32>(__builtin_msa_ilvr_d(
      reinterpret_cast<v2i64>(tmp1), reinterpret_cast<v2i64>(tmp0)));
  result.buf.reg[3] = reinterpret_cast<v4i32>(__builtin_msa_ilvl_d(
      reinterpret_cast<v2i64>(tmp1), reinterpret_cast<v2i64>(tmp0)));
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
      v4i32 t0 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvr_h(src.buf.reg[1], src.buf.reg[0]));
      v4i32 t1 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvr_h(src.buf.reg[3], src.buf.reg[2]));
      v2i64 u0 = reinterpret_cast<v2i64>(__builtin_msa_ilvr_w(t1, t0));
      v2i64 u1 = reinterpret_cast<v2i64>(__builtin_msa_ilvl_w(t1, t0));
      // top-right 4x4
      v4i32 t2 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvr_h(src.buf.reg[5], src.buf.reg[4]));
      v4i32 t3 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvr_h(src.buf.reg[7], src.buf.reg[6]));
      v2i64 u2 = reinterpret_cast<v2i64>(__builtin_msa_ilvr_w(t3, t2));
      v2i64 u3 = reinterpret_cast<v2i64>(__builtin_msa_ilvl_w(t3, t2));
      // bottom-left 4x4
      v4i32 t4 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvl_h(src.buf.reg[1], src.buf.reg[0]));
      v4i32 t5 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvl_h(src.buf.reg[3], src.buf.reg[2]));
      v2i64 u4 = reinterpret_cast<v2i64>(__builtin_msa_ilvr_w(t5, t4));
      v2i64 u5 = reinterpret_cast<v2i64>(__builtin_msa_ilvl_w(t5, t4));
      // bottom-right 4x4
      v4i32 t6 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvl_h(src.buf.reg[5], src.buf.reg[4]));
      v4i32 t7 = reinterpret_cast<v4i32>(
          __builtin_msa_ilvl_h(src.buf.reg[7], src.buf.reg[6]));
      v2i64 u6 = reinterpret_cast<v2i64>(__builtin_msa_ilvr_w(t7, t6));
      v2i64 u7 = reinterpret_cast<v2i64>(__builtin_msa_ilvl_w(t7, t6));

      StoreInt16x8(dst->data(row + 0, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvr_d(u2, u0)));
      StoreInt16x8(dst->data(row + 1, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvl_d(u2, u0)));
      StoreInt16x8(dst->data(row + 2, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvr_d(u3, u1)));
      StoreInt16x8(dst->data(row + 3, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvl_d(u3, u1)));
      StoreInt16x8(dst->data(row + 4, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvr_d(u6, u4)));
      StoreInt16x8(dst->data(row + 5, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvl_d(u6, u4)));
      StoreInt16x8(dst->data(row + 6, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvr_d(u7, u5)));
      StoreInt16x8(dst->data(row + 7, col),
                   reinterpret_cast<v8i16>(__builtin_msa_ilvl_d(u7, u5)));
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

// There's no way to express in C++ the desired machine code for
// StoreFinalOutputImpl<RegBlockUint8<8, 4>, DstType> and
// StoreFinalOutputImpl<RegBlockUint8<8, 8>, DstType>.
// Hence, if we can, we use inline assembly, which takes advantage
// of little-endian byte order and specifics of different CPU revisions.
// Note, clang currently can't derive MSA register names from floating-
// point register names and vice versa in inline assembly.
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) && \
    !defined(__clang__)

// Instructions for pointer-sized operands.
#ifdef GEMMLOWP_MIPS_64
#define GEMMLOWP_MIPS_XADDU "daddu"
#define GEMMLOWP_MIPS_XLSA "dlsa"
#else
#define GEMMLOWP_MIPS_XADDU "addu"
#define GEMMLOWP_MIPS_XLSA "lsa"
#endif

// Stores 4 8-byte half-vectors with a stride.
inline void MipsMsaStore4x8(const RegBlockUint8<8, 4>& src,
                            std::uint8_t* dst_ptr, int stride) {
#if (__mips_isa_rev >= 6)
  // Assembly temporaries that will be handily referred to by their names.
  std::uint8_t *dst_ptr1, *dst_ptr2, *dst_ptr3;
  v16i8 vtmp0, vtmp1;
  asm volatile(
    GEMMLOWP_MIPS_XADDU " %[dst_ptr1], %[dst_ptr0], %[stride]\n"
    "ilvl.d               %w[vtmp0], %w[src0], %w[src0]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr2], %[stride], %[dst_ptr0], 1\n"
    "ilvl.d               %w[vtmp1], %w[src1], %w[src1]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr3], %[stride], %[dst_ptr1], 1\n"
    "sdc1                 %[src0], 0(%[dst_ptr0])\n"
    "sdc1                 %[vtmp0], 0(%[dst_ptr1])\n"
    "sdc1                 %[src1], 0(%[dst_ptr2])\n"
    "sdc1                 %[vtmp1], 0(%[dst_ptr3])\n"
    :
    // Outputs.
    [dst_ptr0] "+r"(dst_ptr), [dst_ptr1] "=&r"(dst_ptr1),
    [dst_ptr2] "=&r"(dst_ptr2), [dst_ptr3] "=&r"(dst_ptr3),
    [vtmp0] "=&f"(vtmp0), [vtmp1] "=&f"(vtmp1)
    :
    // Inputs.
    [src0] "f"(src.buf.reg[0]), [src1] "f"(src.buf.reg[1]),
    [stride] "r"(stride)
    :
    // Clobbers.
    "memory");
#else
  // Assembly temporaries that will be handily referred to by their names.
  std::uint8_t *dst_ptr1, *dst_ptr2, *dst_ptr3;
  int tmp0, tmp1, tmp2, tmp3;
  asm volatile(
    GEMMLOWP_MIPS_XADDU " %[dst_ptr1], %[dst_ptr0], %[stride]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr2], %[stride], %[dst_ptr0], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr3], %[stride], %[dst_ptr1], 1\n"
    "copy_s.w             %[tmp0], %w[src0][0]\n"
    "copy_s.w             %[tmp1], %w[src0][1]\n"
    "copy_s.w             %[tmp2], %w[src0][2]\n"
    "copy_s.w             %[tmp3], %w[src0][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr0])\n"
    "swl                  %[tmp0], 3(%[dst_ptr0])\n"
    "swr                  %[tmp1], 4(%[dst_ptr0])\n"
    "swl                  %[tmp1], 7(%[dst_ptr0])\n"
    "swr                  %[tmp2], 0(%[dst_ptr1])\n"
    "swl                  %[tmp2], 3(%[dst_ptr1])\n"
    "swr                  %[tmp3], 4(%[dst_ptr1])\n"
    "swl                  %[tmp3], 7(%[dst_ptr1])\n"
    "copy_s.w             %[tmp0], %w[src1][0]\n"
    "copy_s.w             %[tmp1], %w[src1][1]\n"
    "copy_s.w             %[tmp2], %w[src1][2]\n"
    "copy_s.w             %[tmp3], %w[src1][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr2])\n"
    "swl                  %[tmp0], 3(%[dst_ptr2])\n"
    "swr                  %[tmp1], 4(%[dst_ptr2])\n"
    "swl                  %[tmp1], 7(%[dst_ptr2])\n"
    "swr                  %[tmp2], 0(%[dst_ptr3])\n"
    "swl                  %[tmp2], 3(%[dst_ptr3])\n"
    "swr                  %[tmp3], 4(%[dst_ptr3])\n"
    "swl                  %[tmp3], 7(%[dst_ptr3])\n"
    :
    // Outputs.
    [dst_ptr0] "+r"(dst_ptr), [dst_ptr1] "=&r"(dst_ptr1),
    [dst_ptr2] "=&r"(dst_ptr2), [dst_ptr3] "=&r"(dst_ptr3), [tmp0] "=&r"(tmp0),
    [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2), [tmp3] "=&r"(tmp3)
    :
    // Inputs.
    [src0] "f"(src.buf.reg[0]), [src1] "f"(src.buf.reg[1]),
    [stride] "r"(stride)
    :
    // Clobbers.
    "memory");
#endif
}

// Stores 8 4-byte quarter-vectors with a stride.
inline void MipsMsaStore8x4(const RegBlockUint8<4, 8>& src,
                            std::uint8_t* dst_ptr, int stride) {
#if (__mips_isa_rev >= 6)
  // Assembly temporaries that will be handily referred to by their names.
  std::uint8_t *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5,
      *dst_ptr6, *dst_ptr7;
  int tmp1, tmp2, tmp3;
  asm volatile(
    GEMMLOWP_MIPS_XADDU " %[dst_ptr1], %[dst_ptr0], %[stride]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr2], %[stride], %[dst_ptr0], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr4], %[stride], %[dst_ptr0], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr3], %[stride], %[dst_ptr1], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr5], %[stride], %[dst_ptr1], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr6], %[stride], %[dst_ptr2], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr7], %[stride], %[dst_ptr3], 2\n"
    "copy_s.w             %[tmp1], %w[src0][1]\n"
    "copy_s.w             %[tmp2], %w[src0][2]\n"
    "copy_s.w             %[tmp3], %w[src0][3]\n"
    "swc1                 %[src0], 0(%[dst_ptr0])\n"
    "sw                   %[tmp1], 0(%[dst_ptr1])\n"
    "sw                   %[tmp2], 0(%[dst_ptr2])\n"
    "sw                   %[tmp3], 0(%[dst_ptr3])\n"
    "copy_s.w             %[tmp1], %w[src1][1]\n"
    "copy_s.w             %[tmp2], %w[src1][2]\n"
    "copy_s.w             %[tmp3], %w[src1][3]\n"
    "swc1                 %[src1], 0(%[dst_ptr4])\n"
    "sw                   %[tmp1], 0(%[dst_ptr5])\n"
    "sw                   %[tmp2], 0(%[dst_ptr6])\n"
    "sw                   %[tmp3], 0(%[dst_ptr7])\n"
    :
    // Outputs.
    [dst_ptr0] "+r"(dst_ptr), [dst_ptr1] "=&r"(dst_ptr1),
    [dst_ptr2] "=&r"(dst_ptr2), [dst_ptr3] "=&r"(dst_ptr3),
    [dst_ptr4] "=&r"(dst_ptr4), [dst_ptr5] "=&r"(dst_ptr5),
    [dst_ptr6] "=&r"(dst_ptr6), [dst_ptr7] "=&r"(dst_ptr7),
    [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2), [tmp3] "=&r"(tmp3)
    :
    // Inputs.
    [src0] "f"(src.buf.reg[0]), [src1] "f"(src.buf.reg[1]),
    [stride] "r"(stride)
    :
    // Clobbers.
    "memory");
#else
  // Assembly temporaries that will be handily referred to by their names.
  std::uint8_t *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5,
      *dst_ptr6, *dst_ptr7;
  int tmp0, tmp1, tmp2, tmp3;
  asm volatile(
    GEMMLOWP_MIPS_XADDU " %[dst_ptr1], %[dst_ptr0], %[stride]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr2], %[stride], %[dst_ptr0], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr4], %[stride], %[dst_ptr0], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr3], %[stride], %[dst_ptr1], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr5], %[stride], %[dst_ptr1], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr6], %[stride], %[dst_ptr2], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr7], %[stride], %[dst_ptr3], 2\n"
    "copy_s.w             %[tmp0], %w[src0][0]\n"
    "copy_s.w             %[tmp1], %w[src0][1]\n"
    "copy_s.w             %[tmp2], %w[src0][2]\n"
    "copy_s.w             %[tmp3], %w[src0][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr0])\n"
    "swl                  %[tmp0], 3(%[dst_ptr0])\n"
    "swr                  %[tmp1], 0(%[dst_ptr1])\n"
    "swl                  %[tmp1], 3(%[dst_ptr1])\n"
    "swr                  %[tmp2], 0(%[dst_ptr2])\n"
    "swl                  %[tmp2], 3(%[dst_ptr2])\n"
    "swr                  %[tmp3], 0(%[dst_ptr3])\n"
    "swl                  %[tmp3], 3(%[dst_ptr3])\n"
    "copy_s.w             %[tmp0], %w[src1][0]\n"
    "copy_s.w             %[tmp1], %w[src1][1]\n"
    "copy_s.w             %[tmp2], %w[src1][2]\n"
    "copy_s.w             %[tmp3], %w[src1][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr4])\n"
    "swl                  %[tmp0], 3(%[dst_ptr4])\n"
    "swr                  %[tmp1], 0(%[dst_ptr5])\n"
    "swl                  %[tmp1], 3(%[dst_ptr5])\n"
    "swr                  %[tmp2], 0(%[dst_ptr6])\n"
    "swl                  %[tmp2], 3(%[dst_ptr6])\n"
    "swr                  %[tmp3], 0(%[dst_ptr7])\n"
    "swl                  %[tmp3], 3(%[dst_ptr7])\n"
    :
    // Outputs.
    [dst_ptr0] "+r"(dst_ptr), [dst_ptr1] "=&r"(dst_ptr1),
    [dst_ptr2] "=&r"(dst_ptr2), [dst_ptr3] "=&r"(dst_ptr3),
    [dst_ptr4] "=&r"(dst_ptr4), [dst_ptr5] "=&r"(dst_ptr5),
    [dst_ptr6] "=&r"(dst_ptr6), [dst_ptr7] "=&r"(dst_ptr7),
    [tmp0] "=&r"(tmp0), [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2),
    [tmp3] "=&r"(tmp3)
    :
    // Inputs.
    [src0] "f"(src.buf.reg[0]), [src1] "f"(src.buf.reg[1]),
    [stride] "r"(stride)
    :
    // Clobbers.
    "memory");
#endif
}

// Stores 8 8-byte half-vectors with a stride.
inline void MipsMsaStore8x8(const RegBlockUint8<8, 8>& src,
                            std::uint8_t* dst_ptr, int stride) {
#if (__mips_isa_rev >= 6)
  // Assembly temporaries that will be handily referred to by their names.
  std::uint8_t *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5,
      *dst_ptr6, *dst_ptr7;
  v16i8 vtmp0, vtmp1, vtmp2, vtmp3;
  asm volatile(
    "ilvl.d               %w[vtmp0], %w[src0], %w[src0]\n"
    GEMMLOWP_MIPS_XADDU " %[dst_ptr1], %[dst_ptr0], %[stride]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr2], %[stride], %[dst_ptr0], 1\n"
    "ilvl.d               %w[vtmp1], %w[src1], %w[src1]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr4], %[stride], %[dst_ptr0], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr3], %[stride], %[dst_ptr1], 1\n"
    "ilvl.d               %w[vtmp2], %w[src2], %w[src2]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr5], %[stride], %[dst_ptr1], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr6], %[stride], %[dst_ptr2], 2\n"
    "ilvl.d               %w[vtmp3], %w[src3], %w[src3]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr7], %[stride], %[dst_ptr3], 2\n"
    "sdc1                 %[src0], 0(%[dst_ptr0])\n"
    "sdc1                 %[vtmp0], 0(%[dst_ptr1])\n"
    "sdc1                 %[src1], 0(%[dst_ptr2])\n"
    "sdc1                 %[vtmp1], 0(%[dst_ptr3])\n"
    "sdc1                 %[src2], 0(%[dst_ptr4])\n"
    "sdc1                 %[vtmp2], 0(%[dst_ptr5])\n"
    "sdc1                 %[src3], 0(%[dst_ptr6])\n"
    "sdc1                 %[vtmp3], 0(%[dst_ptr7])\n"
    :
    // Outputs.
    [dst_ptr0] "+r"(dst_ptr), [dst_ptr1] "=&r"(dst_ptr1),
    [dst_ptr2] "=&r"(dst_ptr2), [dst_ptr3] "=&r"(dst_ptr3),
    [dst_ptr4] "=&r"(dst_ptr4), [dst_ptr5] "=&r"(dst_ptr5),
    [dst_ptr6] "=&r"(dst_ptr6), [dst_ptr7] "=&r"(dst_ptr7),
    [vtmp0] "=&f"(vtmp0), [vtmp1] "=&f"(vtmp1), [vtmp2] "=&f"(vtmp2),
    [vtmp3] "=&f"(vtmp3)
    :
    // Inputs.
    [src0] "f"(src.buf.reg[0]), [src1] "f"(src.buf.reg[1]),
    [src2] "f"(src.buf.reg[2]), [src3] "f"(src.buf.reg[3]),
    [stride] "r"(stride)
    :
    // Clobbers.
    "memory");
#else
  // Assembly temporaries that will be handily referred to by their names.
  std::uint8_t *dst_ptr1, *dst_ptr2, *dst_ptr3, *dst_ptr4, *dst_ptr5,
      *dst_ptr6, *dst_ptr7;
  int tmp0, tmp1, tmp2, tmp3;
  asm volatile(
    GEMMLOWP_MIPS_XADDU " %[dst_ptr1], %[dst_ptr0], %[stride]\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr2], %[stride], %[dst_ptr0], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr4], %[stride], %[dst_ptr0], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr3], %[stride], %[dst_ptr1], 1\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr5], %[stride], %[dst_ptr1], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr6], %[stride], %[dst_ptr2], 2\n"
    GEMMLOWP_MIPS_XLSA  " %[dst_ptr7], %[stride], %[dst_ptr3], 2\n"
    "copy_s.w             %[tmp0], %w[src0][0]\n"
    "copy_s.w             %[tmp1], %w[src0][1]\n"
    "copy_s.w             %[tmp2], %w[src0][2]\n"
    "copy_s.w             %[tmp3], %w[src0][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr0])\n"
    "swl                  %[tmp0], 3(%[dst_ptr0])\n"
    "swr                  %[tmp1], 4(%[dst_ptr0])\n"
    "swl                  %[tmp1], 7(%[dst_ptr0])\n"
    "swr                  %[tmp2], 0(%[dst_ptr1])\n"
    "swl                  %[tmp2], 3(%[dst_ptr1])\n"
    "swr                  %[tmp3], 4(%[dst_ptr1])\n"
    "swl                  %[tmp3], 7(%[dst_ptr1])\n"
    "copy_s.w             %[tmp0], %w[src1][0]\n"
    "copy_s.w             %[tmp1], %w[src1][1]\n"
    "copy_s.w             %[tmp2], %w[src1][2]\n"
    "copy_s.w             %[tmp3], %w[src1][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr2])\n"
    "swl                  %[tmp0], 3(%[dst_ptr2])\n"
    "swr                  %[tmp1], 4(%[dst_ptr2])\n"
    "swl                  %[tmp1], 7(%[dst_ptr2])\n"
    "swr                  %[tmp2], 0(%[dst_ptr3])\n"
    "swl                  %[tmp2], 3(%[dst_ptr3])\n"
    "swr                  %[tmp3], 4(%[dst_ptr3])\n"
    "swl                  %[tmp3], 7(%[dst_ptr3])\n"
    "copy_s.w             %[tmp0], %w[src2][0]\n"
    "copy_s.w             %[tmp1], %w[src2][1]\n"
    "copy_s.w             %[tmp2], %w[src2][2]\n"
    "copy_s.w             %[tmp3], %w[src2][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr4])\n"
    "swl                  %[tmp0], 3(%[dst_ptr4])\n"
    "swr                  %[tmp1], 4(%[dst_ptr4])\n"
    "swl                  %[tmp1], 7(%[dst_ptr4])\n"
    "swr                  %[tmp2], 0(%[dst_ptr5])\n"
    "swl                  %[tmp2], 3(%[dst_ptr5])\n"
    "swr                  %[tmp3], 4(%[dst_ptr5])\n"
    "swl                  %[tmp3], 7(%[dst_ptr5])\n"
    "copy_s.w             %[tmp0], %w[src3][0]\n"
    "copy_s.w             %[tmp1], %w[src3][1]\n"
    "copy_s.w             %[tmp2], %w[src3][2]\n"
    "copy_s.w             %[tmp3], %w[src3][3]\n"
    "swr                  %[tmp0], 0(%[dst_ptr6])\n"
    "swl                  %[tmp0], 3(%[dst_ptr6])\n"
    "swr                  %[tmp1], 4(%[dst_ptr6])\n"
    "swl                  %[tmp1], 7(%[dst_ptr6])\n"
    "swr                  %[tmp2], 0(%[dst_ptr7])\n"
    "swl                  %[tmp2], 3(%[dst_ptr7])\n"
    "swr                  %[tmp3], 4(%[dst_ptr7])\n"
    "swl                  %[tmp3], 7(%[dst_ptr7])\n"
    :
    // Outputs.
    [dst_ptr0] "+r"(dst_ptr), [dst_ptr1] "=&r"(dst_ptr1),
    [dst_ptr2] "=&r"(dst_ptr2), [dst_ptr3] "=&r"(dst_ptr3),
    [dst_ptr4] "=&r"(dst_ptr4), [dst_ptr5] "=&r"(dst_ptr5),
    [dst_ptr6] "=&r"(dst_ptr6), [dst_ptr7] "=&r"(dst_ptr7),
    [tmp0] "=&r"(tmp0), [tmp1] "=&r"(tmp1), [tmp2] "=&r"(tmp2),
    [tmp3] "=&r"(tmp3)
    :
    // Inputs.
    [src0] "f"(src.buf.reg[0]), [src1] "f"(src.buf.reg[1]),
    [src2] "f"(src.buf.reg[2]), [src3] "f"(src.buf.reg[3]),
    [stride] "r"(stride)
    :
    // Clobbers.
    "memory");
#endif
}

#undef GEMMLOWP_MIPS_XADDU
#undef GEMMLOWP_MIPS_XLSA

// Transposes a column-major 8x4 block for storage into a row-major matrix.
inline RegBlockUint8<4, 8> Transpose(const RegBlockUint8<8, 4>& src) {
  v16i8 tmp0 = __builtin_msa_ilvr_b(src.buf.reg[1], src.buf.reg[0]);
  v16i8 tmp1 = __builtin_msa_ilvl_b(src.buf.reg[1], src.buf.reg[0]);
  RegBlockUint8<4, 8> result;
  result.buf.reg[0] = __builtin_msa_ilvr_b(tmp1, tmp0);
  result.buf.reg[1] = __builtin_msa_ilvl_b(tmp1, tmp0);
  return result;
}

inline RegBlockUint8<8, 8> Transpose(const RegBlockUint8<8, 8>& src) {
  v16i8 tmp0[4];
  tmp0[0] = __builtin_msa_ilvr_b(src.buf.reg[1], src.buf.reg[0]);
  tmp0[1] = __builtin_msa_ilvl_b(src.buf.reg[1], src.buf.reg[0]);
  tmp0[2] = __builtin_msa_ilvr_b(src.buf.reg[3], src.buf.reg[2]);
  tmp0[3] = __builtin_msa_ilvl_b(src.buf.reg[3], src.buf.reg[2]);
  v16i8 tmp1[4];
  tmp1[0] = __builtin_msa_ilvr_b(tmp0[1], tmp0[0]);
  tmp1[1] = __builtin_msa_ilvl_b(tmp0[1], tmp0[0]);
  tmp1[2] = __builtin_msa_ilvr_b(tmp0[3], tmp0[2]);
  tmp1[3] = __builtin_msa_ilvl_b(tmp0[3], tmp0[2]);
  RegBlockUint8<8, 8> result;
  result.buf.reg[0] = reinterpret_cast<v16i8>(__builtin_msa_ilvr_w(
      reinterpret_cast<v4i32>(tmp1[2]), reinterpret_cast<v4i32>(tmp1[0])));
  result.buf.reg[1] = reinterpret_cast<v16i8>(__builtin_msa_ilvl_w(
      reinterpret_cast<v4i32>(tmp1[2]), reinterpret_cast<v4i32>(tmp1[0])));
  result.buf.reg[2] = reinterpret_cast<v16i8>(__builtin_msa_ilvr_w(
      reinterpret_cast<v4i32>(tmp1[3]), reinterpret_cast<v4i32>(tmp1[1])));
  result.buf.reg[3] = reinterpret_cast<v16i8>(__builtin_msa_ilvl_w(
      reinterpret_cast<v4i32>(tmp1[3]), reinterpret_cast<v4i32>(tmp1[1])));
  return result;
}

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 4>, DstType> {
  static void Run(const RegBlockUint8<8, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      std::uint8_t* dst_ptr = dst->data(row, col);
      int col_stride = dst->cols_stride();
      MipsMsaStore4x8(src, dst_ptr, col_stride);
    } else {
      const auto& block = Transpose(src);
      std::uint8_t* dst_ptr = dst->data(row, col);
      int row_stride = dst->rows_stride();
      MipsMsaStore8x4(block, dst_ptr, row_stride);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 8>, DstType> {
  static void Run(const RegBlockUint8<8, 8>& src, DstType* dst, int row,
                  int col) {
    const auto& block =
        (DstType::kOrder == MapOrder::ColMajor) ? src : Transpose(src);
    std::uint8_t* dst_ptr = dst->data(row, col);
    int stride = dst->stride();
    MipsMsaStore8x8(block, dst_ptr, stride);
  }
};

#else

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

#endif  // Endianness, compiler.

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_MSA_H_
