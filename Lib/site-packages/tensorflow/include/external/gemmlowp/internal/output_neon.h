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

// output_neon.h: optimized NEON specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_NEON_H_
#define GEMMLOWP_INTERNAL_OUTPUT_NEON_H_

#include "output.h"

#include <arm_neon.h>

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
    int16x4_t res_16 = vqmovn_s32(input.reg[0]);
    uint8x8_t res_8 = vqmovun_s16(vcombine_s16(res_16, res_16));
    output.reg[0] = vget_lane_u32(vreinterpret_u32_u8(res_8), 0);
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
    int16x8_t res_16 =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
    output.reg[0] = vqmovun_s16(res_16);
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
    int16x8_t res_16_0 =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
    int16x8_t res_16_1 =
        vcombine_s16(vqmovn_s32(input.reg[2]), vqmovn_s32(input.reg[3]));
    output.reg[0] = vqmovun_s16(res_16_0);
    output.reg[1] = vqmovun_s16(res_16_1);
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
    int16x8_t res_16[4];
    for (int i = 0; i < 4; i++) {
      res_16[i] = vcombine_s16(vqmovn_s32(input.reg[2 * i]),
                               vqmovn_s32(input.reg[2 * i + 1]));
    }
    for (int i = 0; i < 4; i++) {
      output.reg[i] = vqmovun_s16(res_16[i]);
    }
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt8,
                                 RegBufferInt32<4>> {
  typedef RegBufferInt32<4> InputType;
  typedef RegBufferInt8<4> OutputType;

  typedef OutputStageSaturatingCastToInt8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    int16x4_t res_16 = vqmovn_s32(input.reg[0]);
    int8x8_t res_8 = vqmovn_s16(vcombine_s16(res_16, res_16));
    output.reg[0] = vget_lane_s32(vreinterpret_s32_s8(res_8), 0);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt8,
                                 RegBufferInt32<8>> {
  typedef RegBufferInt32<8> InputType;
  typedef RegBufferInt8<8> OutputType;

  typedef OutputStageSaturatingCastToInt8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    int16x8_t res_16 =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
    output.reg[0] = vqmovn_s16(res_16);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt8,
                                 RegBufferInt32<16>> {
  typedef RegBufferInt32<16> InputType;
  typedef RegBufferInt8<16> OutputType;

  typedef OutputStageSaturatingCastToInt8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    int16x8_t res_16_0 =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
    int16x8_t res_16_1 =
        vcombine_s16(vqmovn_s32(input.reg[2]), vqmovn_s32(input.reg[3]));
    output.reg[0] = vqmovn_s16(res_16_0);
    output.reg[1] = vqmovn_s16(res_16_1);
    return output;
  }
};

template <>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt8,
                                 RegBufferInt32<32>> {
  typedef RegBufferInt32<32> InputType;
  typedef RegBufferInt8<32> OutputType;

  typedef OutputStageSaturatingCastToInt8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    int16x8_t res_16[4];
    for (int i = 0; i < 4; i++) {
      res_16[i] = vcombine_s16(vqmovn_s32(input.reg[2 * i]),
                               vqmovn_s32(input.reg[2 * i + 1]));
    }
    for (int i = 0; i < 4; i++) {
      output.reg[i] = vqmovn_s16(res_16[i]);
    }
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
    output.reg[0] = vqmovn_s32(input.reg[0]);
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
    output.reg[0] =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
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
    output.reg[0] =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
    output.reg[1] =
        vcombine_s16(vqmovn_s32(input.reg[2]), vqmovn_s32(input.reg[3]));
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
    output.reg[0] =
        vcombine_s16(vqmovn_s32(input.reg[0]), vqmovn_s32(input.reg[1]));
    output.reg[1] =
        vcombine_s16(vqmovn_s32(input.reg[2]), vqmovn_s32(input.reg[3]));
    output.reg[2] =
        vcombine_s16(vqmovn_s32(input.reg[4]), vqmovn_s32(input.reg[5]));
    output.reg[3] =
        vcombine_s16(vqmovn_s32(input.reg[6]), vqmovn_s32(input.reg[7]));
    return output;
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
      vst1q_lane_s32(dst->data(row + 0, col), src.buf.reg[0], 0);
      vst1q_lane_s32(dst->data(row + 1, col), src.buf.reg[0], 1);
      vst1q_lane_s32(dst->data(row + 2, col), src.buf.reg[0], 2);
      vst1q_lane_s32(dst->data(row + 3, col), src.buf.reg[0], 3);
      vst1q_lane_s32(dst->data(row + 4, col), src.buf.reg[1], 0);
      vst1q_lane_s32(dst->data(row + 5, col), src.buf.reg[1], 1);
      vst1q_lane_s32(dst->data(row + 6, col), src.buf.reg[1], 2);
      vst1q_lane_s32(dst->data(row + 7, col), src.buf.reg[1], 3);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<4, 1>, DstType> {
  static void Run(const RegBlockInt16<4, 1>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      StoreInt16x4(dst->data(row, col), src.buf.reg[0]);
    } else {
      vst1_lane_s16(dst->data(row + 0, col), src.buf.reg[0], 0);
      vst1_lane_s16(dst->data(row + 1, col), src.buf.reg[0], 1);
      vst1_lane_s16(dst->data(row + 2, col), src.buf.reg[0], 2);
      vst1_lane_s16(dst->data(row + 3, col), src.buf.reg[0], 3);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<8, 1>, DstType> {
  static void Run(const RegBlockInt16<8, 1>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      StoreInt16x8(dst->data(row, col), src.buf.reg[0]);
    } else {
      vst1q_lane_s16(dst->data(row + 0, col), src.buf.reg[0], 0);
      vst1q_lane_s16(dst->data(row + 1, col), src.buf.reg[0], 1);
      vst1q_lane_s16(dst->data(row + 2, col), src.buf.reg[0], 2);
      vst1q_lane_s16(dst->data(row + 3, col), src.buf.reg[0], 3);
      vst1q_lane_s16(dst->data(row + 4, col), src.buf.reg[0], 4);
      vst1q_lane_s16(dst->data(row + 5, col), src.buf.reg[0], 5);
      vst1q_lane_s16(dst->data(row + 6, col), src.buf.reg[0], 6);
      vst1q_lane_s16(dst->data(row + 7, col), src.buf.reg[0], 7);
    }
  }
};

inline RegBlockInt32<4, 4> Transpose(const RegBlockInt32<4, 4>& src) {
  const int32x4x2_t t0 = vtrnq_s32(src.buf.reg[0], src.buf.reg[1]);
  const int32x4x2_t t1 = vtrnq_s32(src.buf.reg[2], src.buf.reg[3]);
  RegBlockInt32<4, 4> result;
  result.buf.reg[0] =
      vcombine_s32(vget_low_s32(t0.val[0]), vget_low_s32(t1.val[0]));
  result.buf.reg[1] =
      vcombine_s32(vget_low_s32(t0.val[1]), vget_low_s32(t1.val[1]));
  result.buf.reg[2] =
      vcombine_s32(vget_high_s32(t0.val[0]), vget_high_s32(t1.val[0]));
  result.buf.reg[3] =
      vcombine_s32(vget_high_s32(t0.val[1]), vget_high_s32(t1.val[1]));
  return result;
}

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<4, 4>, DstType> {
  static void Run(const RegBlockInt32<4, 4>& src, DstType* dst, int row,
                  int col) {
    const auto& block =
        DstType::kOrder == MapOrder::ColMajor ? src : Transpose(src);
    std::int32_t* dst_ptr = dst->data(row, col);
    int stride = dst->stride();
    for (int i = 0; i < 4; i++) {
      vst1q_s32(dst_ptr + i * stride, block.buf.reg[i]);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<4, 4>, DstType> {
  static void Run(const RegBlockInt16<4, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      vst1_s16(dst->data(row, col + 0), vget_low_s16(src.buf.reg[0]));
      vst1_s16(dst->data(row, col + 1), vget_high_s16(src.buf.reg[0]));
      vst1_s16(dst->data(row, col + 2), vget_low_s16(src.buf.reg[1]));
      vst1_s16(dst->data(row, col + 3), vget_high_s16(src.buf.reg[1]));
    } else {
      const int16x4x2_t t0 =
          vtrn_s16(vget_low_s16(src.buf.reg[0]), vget_high_s16(src.buf.reg[0]));
      const int16x4x2_t t1 =
          vtrn_s16(vget_low_s16(src.buf.reg[1]), vget_high_s16(src.buf.reg[1]));
      const int32x4x2_t t =
          vtrnq_s32(vreinterpretq_s32_s16(vcombine_s16(t0.val[0], t0.val[1])),
                    vreinterpretq_s32_s16(vcombine_s16(t1.val[0], t1.val[1])));
      vst1_s16(dst->data(row + 0, col),
               vget_low_s16(vreinterpretq_s16_s32(t.val[0])));
      vst1_s16(dst->data(row + 1, col),
               vget_high_s16(vreinterpretq_s16_s32(t.val[0])));
      vst1_s16(dst->data(row + 2, col),
               vget_low_s16(vreinterpretq_s16_s32(t.val[1])));
      vst1_s16(dst->data(row + 3, col),
               vget_high_s16(vreinterpretq_s16_s32(t.val[1])));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<8, 4>, DstType> {
  static void Run(const RegBlockInt32<8, 4>& src, DstType* dst, int row,
                  int col) {
    std::int32_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      int col_stride = dst->cols_stride();
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + i * col_stride + 0, src.buf.reg[2 * i + 0]);
        vst1q_s32(dst_ptr + i * col_stride + 4, src.buf.reg[2 * i + 1]);
      }
    } else {
      int row_stride = dst->rows_stride();
      RegBlockInt32<4, 4> top;
      top.buf.reg[0] = src.buf.reg[0];
      top.buf.reg[1] = src.buf.reg[2];
      top.buf.reg[2] = src.buf.reg[4];
      top.buf.reg[3] = src.buf.reg[6];
      const auto transpose_top = Transpose(top);
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + i * row_stride, transpose_top.buf.reg[i]);
      }
      RegBlockInt32<4, 4> bottom;
      bottom.buf.reg[0] = src.buf.reg[1];
      bottom.buf.reg[1] = src.buf.reg[3];
      bottom.buf.reg[2] = src.buf.reg[5];
      bottom.buf.reg[3] = src.buf.reg[7];
      const auto transpose_bottom = Transpose(bottom);
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + (i + 4) * row_stride, transpose_bottom.buf.reg[i]);
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<8, 4>, DstType> {
  static void Run(const RegBlockInt16<8, 4>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      vst1q_s16(dst->data(row, col + 0), src.buf.reg[0]);
      vst1q_s16(dst->data(row, col + 1), src.buf.reg[1]);
      vst1q_s16(dst->data(row, col + 2), src.buf.reg[2]);
      vst1q_s16(dst->data(row, col + 3), src.buf.reg[3]);
    } else {
      const int16x8x2_t t0 = vtrnq_s16(src.buf.reg[0], src.buf.reg[1]);
      const int16x8x2_t t1 = vtrnq_s16(src.buf.reg[2], src.buf.reg[3]);
      const int32x4x2_t u0 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[0]),
                                       vreinterpretq_s32_s16(t1.val[0]));
      const int32x4x2_t u1 = vtrnq_s32(vreinterpretq_s32_s16(t0.val[1]),
                                       vreinterpretq_s32_s16(t1.val[1]));
      vst1_s16(dst->data(row + 0, col),
               vget_low_s16(vreinterpretq_s16_s32(u0.val[0])));
      vst1_s16(dst->data(row + 1, col),
               vget_low_s16(vreinterpretq_s16_s32(u1.val[0])));
      vst1_s16(dst->data(row + 2, col),
               vget_low_s16(vreinterpretq_s16_s32(u0.val[1])));
      vst1_s16(dst->data(row + 3, col),
               vget_low_s16(vreinterpretq_s16_s32(u1.val[1])));
      vst1_s16(dst->data(row + 4, col),
               vget_high_s16(vreinterpretq_s16_s32(u0.val[0])));
      vst1_s16(dst->data(row + 5, col),
               vget_high_s16(vreinterpretq_s16_s32(u1.val[0])));
      vst1_s16(dst->data(row + 6, col),
               vget_high_s16(vreinterpretq_s16_s32(u0.val[1])));
      vst1_s16(dst->data(row + 7, col),
               vget_high_s16(vreinterpretq_s16_s32(u1.val[1])));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<8, 8>, DstType> {
  static void Run(const RegBlockInt32<8, 8>& src, DstType* dst, int row,
                  int col) {
    std::int32_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      int col_stride = dst->cols_stride();
      for (int i = 0; i < 8; i++) {
        vst1q_s32(dst_ptr + i * col_stride, src.buf.reg[2 * i]);
        vst1q_s32(dst_ptr + i * col_stride + 4, src.buf.reg[2 * i + 1]);
      }
    } else {
      int row_stride = dst->rows_stride();
      RegBlockInt32<4, 4> top_left;
      top_left.buf.reg[0] = src.buf.reg[0];
      top_left.buf.reg[1] = src.buf.reg[2];
      top_left.buf.reg[2] = src.buf.reg[4];
      top_left.buf.reg[3] = src.buf.reg[6];
      const auto transpose_top_left = Transpose(top_left);
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + i * row_stride, transpose_top_left.buf.reg[i]);
      }
      RegBlockInt32<4, 4> bottom_left;
      bottom_left.buf.reg[0] = src.buf.reg[1];
      bottom_left.buf.reg[1] = src.buf.reg[3];
      bottom_left.buf.reg[2] = src.buf.reg[5];
      bottom_left.buf.reg[3] = src.buf.reg[7];
      const auto transpose_bottom_left = Transpose(bottom_left);
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + (i + 4) * row_stride,
                  transpose_bottom_left.buf.reg[i]);
      }
      RegBlockInt32<4, 4> top_right;
      top_right.buf.reg[0] = src.buf.reg[8];
      top_right.buf.reg[1] = src.buf.reg[10];
      top_right.buf.reg[2] = src.buf.reg[12];
      top_right.buf.reg[3] = src.buf.reg[14];
      const auto transpose_top_right = Transpose(top_right);
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + i * row_stride + 4, transpose_top_right.buf.reg[i]);
      }
      RegBlockInt32<4, 4> bottom_right;
      bottom_right.buf.reg[0] = src.buf.reg[9];
      bottom_right.buf.reg[1] = src.buf.reg[11];
      bottom_right.buf.reg[2] = src.buf.reg[13];
      bottom_right.buf.reg[3] = src.buf.reg[15];
      const auto transpose_bottom_right = Transpose(bottom_right);
      for (int i = 0; i < 4; i++) {
        vst1q_s32(dst_ptr + (i + 4) * row_stride + 4,
                  transpose_bottom_right.buf.reg[i]);
      }
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<4, 1>, DstType> {
  static void Run(const RegBlockInt32<4, 1>& src, DstType* dst, int row,
                  int col) {
    std::int32_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      vst1q_s32(dst_ptr, src.buf.reg[0]);
    } else {
      int row_stride = dst->rows_stride();
      vst1q_lane_s32(dst_ptr + 0 * row_stride, src.buf.reg[0], 0);
      vst1q_lane_s32(dst_ptr + 1 * row_stride, src.buf.reg[0], 1);
      vst1q_lane_s32(dst_ptr + 2 * row_stride, src.buf.reg[0], 2);
      vst1q_lane_s32(dst_ptr + 3 * row_stride, src.buf.reg[0], 3);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt32<1, 4>, DstType> {
  static void Run(const RegBlockInt32<1, 4>& src, DstType* dst, int row,
                  int col) {
    std::int32_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::RowMajor) {
      vst1q_s32(dst_ptr, src.buf.reg[0]);
    } else {
      int col_stride = dst->cols_stride();
      vst1q_lane_s32(dst_ptr + 0 * col_stride, src.buf.reg[0], 0);
      vst1q_lane_s32(dst_ptr + 1 * col_stride, src.buf.reg[0], 1);
      vst1q_lane_s32(dst_ptr + 2 * col_stride, src.buf.reg[0], 2);
      vst1q_lane_s32(dst_ptr + 3 * col_stride, src.buf.reg[0], 3);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<1, 4>, DstType> {
  static void Run(const RegBlockInt16<1, 4>& src, DstType* dst, int row,
                  int col) {
    std::int16_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::RowMajor) {
      vst1_s16(dst_ptr, src.buf.reg[0]);
    } else {
      int col_stride = dst->cols_stride();
      vst1_lane_s16(dst_ptr + 0 * col_stride, src.buf.reg[0], 0);
      vst1_lane_s16(dst_ptr + 1 * col_stride, src.buf.reg[0], 1);
      vst1_lane_s16(dst_ptr + 2 * col_stride, src.buf.reg[0], 2);
      vst1_lane_s16(dst_ptr + 3 * col_stride, src.buf.reg[0], 3);
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
struct StoreFinalOutputImpl<RegBlockUint8<1, 4>, DstType> {
  static void Run(const RegBlockUint8<1, 4>& src, DstType* dst, int row,
                  int col) {
    for (int i = 0; i < 4; i++) {
      *dst->data(row, col + i) = (src.buf.reg[0] >> (8 * i));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 1>, DstType> {
  static void Run(const RegBlockUint8<8, 1>& src, DstType* dst, int row,
                  int col) {
    std::uint8_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      vst1_u8(dst_ptr, src.buf.reg[0]);
    } else {
      const int row_stride = dst->rows_stride();
      vst1_lane_u8(dst_ptr + 0 * row_stride, src.buf.reg[0], 0);
      vst1_lane_u8(dst_ptr + 1 * row_stride, src.buf.reg[0], 1);
      vst1_lane_u8(dst_ptr + 2 * row_stride, src.buf.reg[0], 2);
      vst1_lane_u8(dst_ptr + 3 * row_stride, src.buf.reg[0], 3);
      vst1_lane_u8(dst_ptr + 4 * row_stride, src.buf.reg[0], 4);
      vst1_lane_u8(dst_ptr + 5 * row_stride, src.buf.reg[0], 5);
      vst1_lane_u8(dst_ptr + 6 * row_stride, src.buf.reg[0], 6);
      vst1_lane_u8(dst_ptr + 7 * row_stride, src.buf.reg[0], 7);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<4, 4>, DstType> {
  static void Run(const RegBlockUint8<4, 4>& src, DstType* dst, int row,
                  int col) {
    std::uint8_t* dst_ptr = dst->data(row, col);
    const int row_stride = dst->rows_stride();
    const int col_stride = dst->cols_stride();
    for (int i = 0; i < 2; i++) {
      vst1_lane_u8(dst_ptr + 0 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 0);
      vst1_lane_u8(dst_ptr + 1 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 1);
      vst1_lane_u8(dst_ptr + 2 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 2);
      vst1_lane_u8(dst_ptr + 3 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 3);
      vst1_lane_u8(dst_ptr + 0 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 4);
      vst1_lane_u8(dst_ptr + 1 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 5);
      vst1_lane_u8(dst_ptr + 2 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 6);
      vst1_lane_u8(dst_ptr + 3 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 7);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 4>, DstType> {
  static void Run(const RegBlockUint8<8, 4>& src, DstType* dst, int row,
                  int col) {
    std::uint8_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      int col_stride = dst->cols_stride();
      for (int i = 0; i < 4; i++) {
        vst1_u8(dst_ptr + i * col_stride, src.buf.reg[i]);
      }
    } else {
      int row_stride = dst->rows_stride();
      for (int i = 0; i < 4; i++) {
        std::uint8_t* col_ptr = dst_ptr + i;
        vst1_lane_u8(col_ptr + 0 * row_stride, src.buf.reg[i], 0);
        vst1_lane_u8(col_ptr + 1 * row_stride, src.buf.reg[i], 1);
        vst1_lane_u8(col_ptr + 2 * row_stride, src.buf.reg[i], 2);
        vst1_lane_u8(col_ptr + 3 * row_stride, src.buf.reg[i], 3);
        vst1_lane_u8(col_ptr + 4 * row_stride, src.buf.reg[i], 4);
        vst1_lane_u8(col_ptr + 5 * row_stride, src.buf.reg[i], 5);
        vst1_lane_u8(col_ptr + 6 * row_stride, src.buf.reg[i], 6);
        vst1_lane_u8(col_ptr + 7 * row_stride, src.buf.reg[i], 7);
      }
    }
  }
};

inline RegBlockUint8<8, 8> Transpose(const RegBlockUint8<8, 8>& src) {
  uint8x8x2_t a[4];
  a[0] = vtrn_u8(src.buf.reg[0], src.buf.reg[1]);
  a[1] = vtrn_u8(src.buf.reg[2], src.buf.reg[3]);
  a[2] = vtrn_u8(src.buf.reg[4], src.buf.reg[5]);
  a[3] = vtrn_u8(src.buf.reg[6], src.buf.reg[7]);
  uint16x4x2_t b[4];
  b[0] = vtrn_u16(vreinterpret_u16_u8(a[0].val[0]),
                  vreinterpret_u16_u8(a[1].val[0]));
  b[1] = vtrn_u16(vreinterpret_u16_u8(a[0].val[1]),
                  vreinterpret_u16_u8(a[1].val[1]));
  b[2] = vtrn_u16(vreinterpret_u16_u8(a[2].val[0]),
                  vreinterpret_u16_u8(a[3].val[0]));
  b[3] = vtrn_u16(vreinterpret_u16_u8(a[2].val[1]),
                  vreinterpret_u16_u8(a[3].val[1]));
  uint32x2x2_t c[4];
  c[0] = vtrn_u32(vreinterpret_u32_u16(b[0].val[0]),
                  vreinterpret_u32_u16(b[2].val[0]));
  c[1] = vtrn_u32(vreinterpret_u32_u16(b[1].val[0]),
                  vreinterpret_u32_u16(b[3].val[0]));
  c[2] = vtrn_u32(vreinterpret_u32_u16(b[0].val[1]),
                  vreinterpret_u32_u16(b[2].val[1]));
  c[3] = vtrn_u32(vreinterpret_u32_u16(b[1].val[1]),
                  vreinterpret_u32_u16(b[3].val[1]));
  RegBlockUint8<8, 8> result;
  result.buf.reg[0] = vreinterpret_u8_u32(c[0].val[0]);
  result.buf.reg[1] = vreinterpret_u8_u32(c[1].val[0]);
  result.buf.reg[2] = vreinterpret_u8_u32(c[2].val[0]);
  result.buf.reg[3] = vreinterpret_u8_u32(c[3].val[0]);
  result.buf.reg[4] = vreinterpret_u8_u32(c[0].val[1]);
  result.buf.reg[5] = vreinterpret_u8_u32(c[1].val[1]);
  result.buf.reg[6] = vreinterpret_u8_u32(c[2].val[1]);
  result.buf.reg[7] = vreinterpret_u8_u32(c[3].val[1]);
  return result;
}

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockUint8<8, 8>, DstType> {
  static void Run(const RegBlockUint8<8, 8>& src, DstType* dst, int row,
                  int col) {
    const auto& block =
        DstType::kOrder == MapOrder::ColMajor ? src : Transpose(src);
    std::uint8_t* dst_ptr = dst->data(row, col);
    int stride = dst->stride();
    for (int i = 0; i < 8; i++) {
      vst1_u8(dst_ptr + i * stride, block.buf.reg[i]);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt8<4, 1>, DstType> {
  static void Run(const RegBlockInt8<4, 1>& src, DstType* dst, int row,
                  int col) {
    const std::int32_t src_reg = src.buf.reg[0];
    for (int i = 0; i < 4; i++) {
      *dst->data(row + i, col) = (src_reg >> (8 * i));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt8<1, 4>, DstType> {
  static void Run(const RegBlockInt8<1, 4>& src, DstType* dst, int row,
                  int col) {
    for (int i = 0; i < 4; i++) {
      *dst->data(row, col + i) = (src.buf.reg[0] >> (8 * i));
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt8<8, 1>, DstType> {
  static void Run(const RegBlockInt8<8, 1>& src, DstType* dst, int row,
                  int col) {
    std::int8_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      vst1_s8(dst_ptr, src.buf.reg[0]);
    } else {
      const int row_stride = dst->rows_stride();
      vst1_lane_s8(dst_ptr + 0 * row_stride, src.buf.reg[0], 0);
      vst1_lane_s8(dst_ptr + 1 * row_stride, src.buf.reg[0], 1);
      vst1_lane_s8(dst_ptr + 2 * row_stride, src.buf.reg[0], 2);
      vst1_lane_s8(dst_ptr + 3 * row_stride, src.buf.reg[0], 3);
      vst1_lane_s8(dst_ptr + 4 * row_stride, src.buf.reg[0], 4);
      vst1_lane_s8(dst_ptr + 5 * row_stride, src.buf.reg[0], 5);
      vst1_lane_s8(dst_ptr + 6 * row_stride, src.buf.reg[0], 6);
      vst1_lane_s8(dst_ptr + 7 * row_stride, src.buf.reg[0], 7);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt8<4, 4>, DstType> {
  static void Run(const RegBlockInt8<4, 4>& src, DstType* dst, int row,
                  int col) {
    std::int8_t* dst_ptr = dst->data(row, col);
    const int row_stride = dst->rows_stride();
    const int col_stride = dst->cols_stride();
    for (int i = 0; i < 2; i++) {
      vst1_lane_s8(dst_ptr + 0 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 0);
      vst1_lane_s8(dst_ptr + 1 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 1);
      vst1_lane_s8(dst_ptr + 2 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 2);
      vst1_lane_s8(dst_ptr + 3 * row_stride + (2 * i + 0) * col_stride,
                   src.buf.reg[i], 3);
      vst1_lane_s8(dst_ptr + 0 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 4);
      vst1_lane_s8(dst_ptr + 1 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 5);
      vst1_lane_s8(dst_ptr + 2 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 6);
      vst1_lane_s8(dst_ptr + 3 * row_stride + (2 * i + 1) * col_stride,
                   src.buf.reg[i], 7);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt8<8, 4>, DstType> {
  static void Run(const RegBlockInt8<8, 4>& src, DstType* dst, int row,
                  int col) {
    std::int8_t* dst_ptr = dst->data(row, col);
    if (DstType::kOrder == MapOrder::ColMajor) {
      int col_stride = dst->cols_stride();
      for (int i = 0; i < 4; i++) {
        vst1_s8(dst_ptr + i * col_stride, src.buf.reg[i]);
      }
    } else {
      int row_stride = dst->rows_stride();
      for (int i = 0; i < 4; i++) {
        std::int8_t* col_ptr = dst_ptr + i;
        vst1_lane_s8(col_ptr + 0 * row_stride, src.buf.reg[i], 0);
        vst1_lane_s8(col_ptr + 1 * row_stride, src.buf.reg[i], 1);
        vst1_lane_s8(col_ptr + 2 * row_stride, src.buf.reg[i], 2);
        vst1_lane_s8(col_ptr + 3 * row_stride, src.buf.reg[i], 3);
        vst1_lane_s8(col_ptr + 4 * row_stride, src.buf.reg[i], 4);
        vst1_lane_s8(col_ptr + 5 * row_stride, src.buf.reg[i], 5);
        vst1_lane_s8(col_ptr + 6 * row_stride, src.buf.reg[i], 6);
        vst1_lane_s8(col_ptr + 7 * row_stride, src.buf.reg[i], 7);
      }
    }
  }
};

inline RegBlockInt8<8, 8> Transpose(const RegBlockInt8<8, 8>& src) {
  int8x8x2_t a[4];
  a[0] = vtrn_s8(src.buf.reg[0], src.buf.reg[1]);
  a[1] = vtrn_s8(src.buf.reg[2], src.buf.reg[3]);
  a[2] = vtrn_s8(src.buf.reg[4], src.buf.reg[5]);
  a[3] = vtrn_s8(src.buf.reg[6], src.buf.reg[7]);
  int16x4x2_t b[4];
  b[0] = vtrn_s16(vreinterpret_s16_s8(a[0].val[0]),
                  vreinterpret_s16_s8(a[1].val[0]));
  b[1] = vtrn_s16(vreinterpret_s16_s8(a[0].val[1]),
                  vreinterpret_s16_s8(a[1].val[1]));
  b[2] = vtrn_s16(vreinterpret_s16_s8(a[2].val[0]),
                  vreinterpret_s16_s8(a[3].val[0]));
  b[3] = vtrn_s16(vreinterpret_s16_s8(a[2].val[1]),
                  vreinterpret_s16_s8(a[3].val[1]));
  int32x2x2_t c[4];
  c[0] = vtrn_s32(vreinterpret_s32_s16(b[0].val[0]),
                  vreinterpret_s32_s16(b[2].val[0]));
  c[1] = vtrn_s32(vreinterpret_s32_s16(b[1].val[0]),
                  vreinterpret_s32_s16(b[3].val[0]));
  c[2] = vtrn_s32(vreinterpret_s32_s16(b[0].val[1]),
                  vreinterpret_s32_s16(b[2].val[1]));
  c[3] = vtrn_s32(vreinterpret_s32_s16(b[1].val[1]),
                  vreinterpret_s32_s16(b[3].val[1]));
  RegBlockInt8<8, 8> result;
  result.buf.reg[0] = vreinterpret_s8_s32(c[0].val[0]);
  result.buf.reg[1] = vreinterpret_s8_s32(c[1].val[0]);
  result.buf.reg[2] = vreinterpret_s8_s32(c[2].val[0]);
  result.buf.reg[3] = vreinterpret_s8_s32(c[3].val[0]);
  result.buf.reg[4] = vreinterpret_s8_s32(c[0].val[1]);
  result.buf.reg[5] = vreinterpret_s8_s32(c[1].val[1]);
  result.buf.reg[6] = vreinterpret_s8_s32(c[2].val[1]);
  result.buf.reg[7] = vreinterpret_s8_s32(c[3].val[1]);
  return result;
}

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt8<8, 8>, DstType> {
  static void Run(const RegBlockInt8<8, 8>& src, DstType* dst, int row,
                  int col) {
    const auto& block =
        DstType::kOrder == MapOrder::ColMajor ? src : Transpose(src);
    std::int8_t* dst_ptr = dst->data(row, col);
    int stride = dst->stride();
    for (int i = 0; i < 8; i++) {
      vst1_s8(dst_ptr + i * stride, block.buf.reg[i]);
    }
  }
};

template <typename DstType>
struct StoreFinalOutputImpl<RegBlockInt16<8, 8>, DstType> {
  static void Run(const RegBlockInt16<8, 8>& src, DstType* dst, int row,
                  int col) {
    if (DstType::kOrder == MapOrder::ColMajor) {
      vst1q_s16(dst->data(row, col + 0), src.buf.reg[0]);
      vst1q_s16(dst->data(row, col + 1), src.buf.reg[1]);
      vst1q_s16(dst->data(row, col + 2), src.buf.reg[2]);
      vst1q_s16(dst->data(row, col + 3), src.buf.reg[3]);
      vst1q_s16(dst->data(row, col + 4), src.buf.reg[4]);
      vst1q_s16(dst->data(row, col + 5), src.buf.reg[5]);
      vst1q_s16(dst->data(row, col + 6), src.buf.reg[6]);
      vst1q_s16(dst->data(row, col + 7), src.buf.reg[7]);
    } else {
      int16x8x2_t a[4];
      a[0] = vtrnq_s16(src.buf.reg[0], src.buf.reg[1]);
      a[1] = vtrnq_s16(src.buf.reg[2], src.buf.reg[3]);
      a[2] = vtrnq_s16(src.buf.reg[4], src.buf.reg[5]);
      a[3] = vtrnq_s16(src.buf.reg[6], src.buf.reg[7]);
      int32x4x2_t b[4];
      b[0] = vtrnq_s32(vreinterpretq_s32_s16(a[0].val[0]),
                       vreinterpretq_s32_s16(a[1].val[0]));
      b[1] = vtrnq_s32(vreinterpretq_s32_s16(a[0].val[1]),
                       vreinterpretq_s32_s16(a[1].val[1]));
      b[2] = vtrnq_s32(vreinterpretq_s32_s16(a[2].val[0]),
                       vreinterpretq_s32_s16(a[3].val[0]));
      b[3] = vtrnq_s32(vreinterpretq_s32_s16(a[2].val[1]),
                       vreinterpretq_s32_s16(a[3].val[1]));
      vst1_s16(dst->data(row + 0, col + 0),
               vget_low_s16(vreinterpretq_s16_s32(b[0].val[0])));
      vst1_s16(dst->data(row + 0, col + 4),
               vget_low_s16(vreinterpretq_s16_s32(b[2].val[0])));
      vst1_s16(dst->data(row + 1, col + 0),
               vget_low_s16(vreinterpretq_s16_s32(b[1].val[0])));
      vst1_s16(dst->data(row + 1, col + 4),
               vget_low_s16(vreinterpretq_s16_s32(b[3].val[0])));
      vst1_s16(dst->data(row + 2, col + 0),
               vget_low_s16(vreinterpretq_s16_s32(b[0].val[1])));
      vst1_s16(dst->data(row + 2, col + 4),
               vget_low_s16(vreinterpretq_s16_s32(b[2].val[1])));
      vst1_s16(dst->data(row + 3, col + 0),
               vget_low_s16(vreinterpretq_s16_s32(b[1].val[1])));
      vst1_s16(dst->data(row + 3, col + 4),
               vget_low_s16(vreinterpretq_s16_s32(b[3].val[1])));
      vst1_s16(dst->data(row + 4, col + 0),
               vget_high_s16(vreinterpretq_s16_s32(b[0].val[0])));
      vst1_s16(dst->data(row + 4, col + 4),
               vget_high_s16(vreinterpretq_s16_s32(b[2].val[0])));
      vst1_s16(dst->data(row + 5, col + 0),
               vget_high_s16(vreinterpretq_s16_s32(b[1].val[0])));
      vst1_s16(dst->data(row + 5, col + 4),
               vget_high_s16(vreinterpretq_s16_s32(b[3].val[0])));
      vst1_s16(dst->data(row + 6, col + 0),
               vget_high_s16(vreinterpretq_s16_s32(b[0].val[1])));
      vst1_s16(dst->data(row + 6, col + 4),
               vget_high_s16(vreinterpretq_s16_s32(b[2].val[1])));
      vst1_s16(dst->data(row + 7, col + 0),
               vget_high_s16(vreinterpretq_s16_s32(b[1].val[1])));
      vst1_s16(dst->data(row + 7, col + 4),
               vget_high_s16(vreinterpretq_s16_s32(b[3].val[1])));
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_NEON_H_
