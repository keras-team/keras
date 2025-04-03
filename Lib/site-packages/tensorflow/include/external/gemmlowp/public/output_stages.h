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

// output_stages.h: public definitions of the output stages that can
// be assembled into an output pipeline, to control how internal
// 32-bit accumulators are transformed to obtain the final uint8
// result matrix entries.

#ifndef GEMMLOWP_PUBLIC_OUTPUT_STAGES_H_
#define GEMMLOWP_PUBLIC_OUTPUT_STAGES_H_

#include <tuple>

#include "../internal/common.h"

namespace gemmlowp {

// This output stage takes int32 values and returns still int32 values,
// but "quantized down" to the uint8 scale; in other words, its output
// is typically what one would then clamp to [0..255] and cast to uint8
// (see OutputStageSaturatingCastToUint8).
//
// This "quantization down" process depends on 3 parameters,
//   result_offset, result_mult_int, result_shift,
// and the result is:
//   ((input + result_offset) * result_mult_int + rounding) >> result_shift
// where
//   rounding = (result_shift < 1) ? 0 : (1 << (result_shift - 1));
struct OutputStageQuantizeDownInt32ToUint8Scale {
  std::int32_t result_offset;
  std::int32_t result_mult_int;
  std::int32_t result_shift;
};

// This output stage takes int32 values and returns still int32 values,
// but "quantized down" to the uint8 scale; in other words, its output
// is typically what one would then clamp to [0..255] and cast to uint8
// (see OutputStageSaturatingCastToUint8).
//
// This "quantization down" process depends on 3 parameters,
//   result_offset, result_mult_int, result_shift,
// and the result is:
//   ((input + result_offset) * result_mult_int + rounding) >> result_shift
// where
//   rounding = (result_shift < 1) ? 0 : (1 << (result_shift - 1));
//
// Difference from OutputStageQuantizeDownInt32ToUint8Scale here is that each
// row or column of the output (depending on tShape) has its own result_offset
// and result_mult_int numbers.
template <VectorShape tShape>
struct OutputStageQuantizeDownInt32ToUint8ScalePC {
  VectorMap<const std::int32_t, tShape> result_offset;
  VectorMap<const std::int32_t, tShape> result_mult_int;
  std::int32_t result_shift;
};

// This output stage takes int32 values and returns still int32 values,
// but "quantized down" to a difference scale; for example, in a pipeline
// that outputs uint8 values in [0..255], the output of this stage would be
// int32 values ready to be clamped to [0..255] and casted to uint8
// (see OutputStageSaturatingCastToUint8).
//
// This "quantization down" process depends on 3 parameters,
//   result_offset, result_fixedpoint_multiplier, result_shift,
// and the result is:
//   ((FixedPointMul(input, result_fixedpoint_multiplier) +
//   rounding) >> result_shift) + result_offset_after_shift
// where
//   rounding = (result_shift < 1) ? 0 : (1 << (result_shift - 1));
// and where FixedPointMul(x, y) is the nearest integer to the following
// mathematical expression, evaluated without overflow or intermediate
// rounding:
//   (x * y) / 2^31
// In practice, it is expected that FixedPointMul will be implemented
// using hardware "rounding doubling int32 multiply high" instructions,
// such as VQRDMULH on ARM. See in fixedpoint.h the generic function,
// SaturatingRoundingDoublingHighMul.
//
// Notice that the other difference from
// OutputStageQuantizeDownInt32ToUint8Scale is that the result offset
// is applied after the multiplier and shift, not before. This ensures
// that no matter what the multiplier and shift are, the result offset
// is effectively integral: offsetting the final result by an integer.
// The motivation for this is to faithfully support quantization schemes
// where the formula linking quantized values to the real mathematical
// values that they represent, is of the form
//
//   real_value = scale * (quantized_value - zero_point)
//
// where scale is a real number (represented in quantized form by
// result_fixedpoint_multiplier and result_shift) and zero_point
// is an integer telling which quantized value correspond to the
// real value 0, and is represented here by (the opposite of)
// result_offset_after_shift.
// The motivation for such a quantization scheme, designed to
// ensure that 0 is always a representable value, is that in
// many applications, we need to 0-pad arrays and that can only be
// done for quantized arrays if 0 is a representable value in
// quantized form. In particular, convolution-like operations
// are often implemented using 0-padding, or "im2col"-like
// expansions that implicitly rely on 0-padding. If 0 were not
// a representable value, such operations would have to pad
// using a nonzero value, introducing bias in the computation.
struct OutputStageQuantizeDownInt32ByFixedPoint {
  std::int32_t result_fixedpoint_multiplier;
  std::int32_t result_shift;
  std::int32_t result_offset_after_shift;
};

// OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint is the old deprecated
// name of OutputStageQuantizeDownInt32ByFixedPoint, before we noticed that
// there really wasn't anything Uint8-specific about it.
using OutputStageQuantizeDownInt32ToUint8ScaleByFixedPoint = OutputStageQuantizeDownInt32ByFixedPoint;

// Variant of OutputStageQuantizeDownInt32ByFixedPoint where the 'shift'
// is not necessarily just a right shift, so we can represent multipliers
// greater than 1. This takes an result_exponent parameter; when it's
// <= 0, this is equivalent to OutputStageQuantizeDownInt32ByFixedPoint
// with result_shift = -result_exponent.
// In the general case, this consists in first left-shifting by
// std::max(result_exponent, 0), before doing the same as
// OutputStageQuantizeDownInt32ByFixedPoint with
// result_shift = std::max(-result_exponent, 0).
struct OutputStageScaleInt32ByFixedPointAndExponent {
  std::int32_t result_fixedpoint_multiplier;
  std::int32_t result_exponent;
  std::int32_t result_offset_after_shift;
};

// Variant of OutputStageQuantizeDownInt32ByFixedPoint where the 'shift'
// is not necessarily just a right shift, so we can represent multipliers
// greater than 1. This takes an result_exponent parameter; when it's
// <= 0, this is equivalent to OutputStageQuantizeDownInt32ByFixedPoint
// with result_shift = -result_exponent.
// In the general case, this consists in first left-shifting by
// std::max(result_exponent, 0), before doing the same as
// OutputStageQuantizeDownInt32ByFixedPoint with
// result_shift = std::max(-result_exponent, 0).
//
// Difference from OutputStageScaleInt32ByFixedPointAndExponent here is that
// each row or column of the output (depending on tShape) has its own
// result_fixedpoint_multiplier and result_exponent numbers.
template <VectorShape tShape>
struct OutputStageScaleInt32ByFixedPointAndExponentPC {
  VectorMap<const std::int32_t, tShape> result_fixedpoint_multiplier;
  VectorMap<const std::int32_t, tShape> result_exponent;
  std::int32_t result_offset_after_shift;
};

// This output stage takes int32 values that are expected to be already
// on the final uint8 scale, but not necessarily in the [0..255] range.
// It clamps them to the [0..255] range and returns them casted to uint8.
struct OutputStageSaturatingCastToUint8 {};

// This output stage takes int32 values that are expected to be already
// on the final int8 scale, but not necessarily in the [-128..127] range.
// It clamps them to the [-128..127] range and returns them casted to int8.
struct OutputStageSaturatingCastToInt8 {};

// This output stage takes int32 values that are expected to be already
// in the [0..255] range and returns them casted to uint8.
// This stage can save time if used instead of the
// OutputStageSaturatingCastToUint8 stage immediately after the
// OutputStageClamp stage.
struct OutputStageTruncatingCastToUint8 {};

// This output stage takes int32 values that are expected to be already
// on the final int16 scale, but not necessarily in the [-32768..32767] range.
// It clamps them to the [-32768..32767] range and returns them casted to int16.
struct OutputStageSaturatingCastToInt16 {};

// This output stage depends on a "bias vector" that should contain int32
// entries, and be either a row-vector of the same number of columns as the
// result matrix, or a column-vector of the same number of rows as the
// result matrix. This output stage takes int32 values and adds to them
// the corresponding entry of the bias vector (broadcasted in the other
// direction to fit the matrix's shape), outputting int32 values.
template <typename VectorType>
struct OutputStageBiasAddition {
  VectorType bias_vector;
};

// This output stage clamps value between the specified min and max bounds.
// It can be used to implement "rectified linear unit" activation functions
// in neural networks.
struct OutputStageClamp {
  std::int32_t min;
  std::int32_t max;
};

struct OutputStageTanh {
  std::int32_t real_zero_as_int32;
  std::int32_t real_amplitude_as_int32;
};

// An output pipeline is just a std::tuple of output stages.
// This function generates a standard output pipeline consisting of two stages:
// OutputStageQuantizeDownInt32ToUint8Scale, OutputStageSaturatingCastToUint8.
inline std::tuple<OutputStageQuantizeDownInt32ToUint8Scale,
                  OutputStageSaturatingCastToUint8>
MakeStandardOutputPipeline(std::int32_t result_offset,
                           std::int32_t result_mult_int,
                           std::int32_t result_shift) {
  OutputStageQuantizeDownInt32ToUint8Scale quantize_down_stage;
  quantize_down_stage.result_offset = result_offset;
  quantize_down_stage.result_mult_int = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  OutputStageSaturatingCastToUint8 saturating_cast_stage;
  return std::make_tuple(quantize_down_stage, saturating_cast_stage);
}

// An output pipeline is just a std::tuple of output stages.
// This function generates a standard output pipeline consisting of two stages:
// OutputStageQuantizeDownInt32ToUint8ScalePC, OutputStageSaturatingCastToUint8.
template <VectorShape tShape>
inline std::tuple<OutputStageQuantizeDownInt32ToUint8ScalePC<tShape>,
                  OutputStageSaturatingCastToUint8>
MakeStandardOutputPipeline(
    const VectorMap<const std::int32_t, tShape>& result_offset,
    const VectorMap<const std::int32_t, tShape>& result_mult_int,
    std::int32_t result_shift) {
  OutputStageQuantizeDownInt32ToUint8ScalePC<tShape> quantize_down_stage;
  quantize_down_stage.result_offset = result_offset;
  quantize_down_stage.result_mult_int = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  OutputStageSaturatingCastToUint8 saturating_cast_stage;
  return std::make_tuple(quantize_down_stage, saturating_cast_stage);
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_OUTPUT_STAGES_H_
