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

// output.h: processing the 32-bit accumulators output by the unpack
// stage, obtaining the final result matrix entries and storing them into
// the destination matrix.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_H_
#define GEMMLOWP_INTERNAL_OUTPUT_H_

#include <cmath>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include "../fixedpoint/fixedpoint.h"
#include "../public/output_stages.h"
#include "simd_wrappers.h"

namespace gemmlowp {

template <typename OutputStage, typename InputBufferType>
struct OutputStageEvalBufferImpl {
  // This generic template body should never be hit.
  static_assert(
      std::is_same<InputBufferType, void>::value,
      "Unimplemented: missing implementation of this output pipeline stage "
      "for this data type. This would happen if some architecture-specific "
      "SIMD back-end (output_$arch.h) were incomplete.");
};

template <typename OutputStage, typename InputType>
struct OutputStageEvalImpl {
  static constexpr int kRows = InputType::kRows;
  static constexpr int kCols = InputType::kCols;
  using InputBufferType = typename InputType::BufferType;
  using BufferEvalImplType =
      OutputStageEvalBufferImpl<OutputStage, InputBufferType>;
  using OutputBufferType = typename BufferEvalImplType::OutputType;
  using OutputScalarType = typename OutputBufferType::ScalarType;
  using OutputType = RegisterBlock<OutputScalarType, kRows, kCols>;

  OutputStageEvalImpl(const OutputStage& s) : buffer_eval_impl(s) {}

  OutputType Eval(InputType input, int, int) const {
    OutputType output;
    output.buf = buffer_eval_impl.Eval(input.buf);
    return output;
  }

  const BufferEvalImplType buffer_eval_impl;
};

template <int Size>
struct OutputStageEvalBufferImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                                 RegisterBuffer<std::int32_t, Size>> {
  using InputType = RegisterBuffer<std::int32_t, Size>;
  using OutputType = RegisterBuffer<std::int32_t, Size>;

  typedef OutputStageQuantizeDownInt32ToUint8Scale OutputStage;

  OutputStageEvalBufferImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input) const {
    const int result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int;
    using RegisterType = typename InputType::RegisterType;
    const RegisterType result_offset =
        Dup<RegisterType>(output_stage.result_offset);
    OutputType output;
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      output.reg[i] = RoundingDivideByPOT(
          Mul(Add(input.reg[i], result_offset), result_mult_int), result_shift);
    }
    return output;
  }

  const OutputStage& output_stage;
};

template <int Rows, int Cols, VectorShape Shape>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8ScalePC<Shape>,
                           RegisterBlock<std::int32_t, Rows, Cols>> {
  typedef RegisterBlock<std::int32_t, Rows, Cols> InputType;
  typedef RegisterBlock<std::int32_t, Rows, Cols> OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<Shape> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    OutputType output;
    const int result_shift = output_stage.result_shift;
    const int pos = Shape == VectorShape::Col ? row : col;
    const auto result_mult_int =
        LoadForBroadcasting<InputType>(output_stage.result_mult_int, pos);
    const auto result_offset =
        LoadForBroadcasting<InputType>(output_stage.result_offset, pos);
    const auto dividend = BroadcastMul<InputType>(
        BroadcastAdd<InputType>(input, result_offset), result_mult_int);
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      output.buf.reg[i] =
          RoundingDivideByPOT(dividend.buf.reg[i], result_shift);
    }
    return output;
  }

  const OutputStage& output_stage;
};

template <int Size>
struct OutputStageEvalBufferImpl<
    OutputStageQuantizeDownInt32ByFixedPoint,
    RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::int32_t, Size> OutputType;

  typedef OutputStageQuantizeDownInt32ByFixedPoint OutputStage;

  OutputStageEvalBufferImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    using RegisterType = typename InputType::RegisterType;
    const RegisterType result_offset_after_shift =
        Dup<RegisterType>(output_stage.result_offset_after_shift);
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      const RegisterType mulhigh_val = SaturatingRoundingDoublingHighMul(
          input.reg[i], output_stage.result_fixedpoint_multiplier);
      output.reg[i] =
          Add(RoundingDivideByPOT(mulhigh_val, output_stage.result_shift),
              result_offset_after_shift);
    }
    return output;
  }

  const OutputStage& output_stage;
};

template <int Size>
struct OutputStageEvalBufferImpl<OutputStageScaleInt32ByFixedPointAndExponent,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::int32_t, Size> OutputType;

  typedef OutputStageScaleInt32ByFixedPointAndExponent OutputStage;

  OutputStageEvalBufferImpl(const OutputStage& s) : output_stage(s) {
    left_shift = std::max(0, output_stage.result_exponent);
    right_shift = std::max(0, -output_stage.result_exponent);
  }

  OutputType Eval(InputType input) const {
    OutputType output;
    using RegisterType = typename InputType::RegisterType;
    const RegisterType result_offset_after_shift =
        Dup<RegisterType>(output_stage.result_offset_after_shift);
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      const RegisterType mulhigh_val = SaturatingRoundingDoublingHighMul(
          ShiftLeft(input.reg[i], left_shift),
          output_stage.result_fixedpoint_multiplier);
      output.reg[i] = Add(RoundingDivideByPOT(mulhigh_val, right_shift),
                          result_offset_after_shift);
    }
    return output;
  }

  const OutputStage& output_stage;
  int left_shift;
  int right_shift;
};

template <int Rows, int Cols, VectorShape Shape>
struct OutputStageEvalImpl<
    OutputStageScaleInt32ByFixedPointAndExponentPC<Shape>,
    RegisterBlock<std::int32_t, Rows, Cols>> {
  typedef RegisterBlock<std::int32_t, Rows, Cols> InputType;
  typedef RegisterBlock<std::int32_t, Rows, Cols> OutputType;

  typedef OutputStageScaleInt32ByFixedPointAndExponentPC<Shape> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    OutputType output;
    const int pos = Shape == VectorShape::Row ? col : row;
    using RegisterType = typename InputType::RegisterType;
    const RegisterType result_offset_after_shift =
        Dup<RegisterType>(output_stage.result_offset_after_shift);
    auto left_shift =
        LoadForBroadcasting<InputType>(output_stage.result_exponent, pos);
    auto right_shift =
        LoadForBroadcasting<InputType>(output_stage.result_exponent, pos);
    const auto result_fixedpoint_multiplier = LoadForBroadcasting<InputType>(
        output_stage.result_fixedpoint_multiplier, pos);
    for (int i = 0; i < decltype(left_shift)::kRegisterCount; i++) {
      left_shift.buf.reg[i] = Max(left_shift.buf.reg[i], 0);
      right_shift.buf.reg[i] = Max(-right_shift.buf.reg[i], 0);
    }
    const auto mulhigh_val = BroadcastSaturatingRoundingDoublingHighMul(
        BroadcastShiftLeft(input, left_shift), result_fixedpoint_multiplier);
    const auto rdpot_val =
        BroadcastRoundingDivideByPOT(mulhigh_val, right_shift);
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      output.buf.reg[i] = Add(rdpot_val.buf.reg[i], result_offset_after_shift);
    }
    return output;
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageSaturatingCastToUint8 for scalar data.
template <int Size>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToUint8,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::uint8_t, Size> OutputType;
  static_assert(InputType::kRegisterLanes == 1,
                "This path is only for scalar values");

  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      std::int32_t data = input.reg[i];
      output.reg[i] = data > 255 ? 255 : data < 0 ? 0 : data;
    }
    return output;
  }
};

// Implementation of OutputStageSaturatingCastToInt8 for scalar data.
template <int Size>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt8,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::int8_t, Size> OutputType;
  static_assert(InputType::kRegisterLanes == 1,
                "This path is only for scalar values");

  typedef OutputStageSaturatingCastToInt8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      std::int32_t data = input.reg[i];
      output.reg[i] = data > 127 ? 127 : data < -128 ? -128 : data;
    }
    return output;
  }
};

// Implementation of OutputStageSaturatingCastToInt16 for scalar data.
template <int Size>
struct OutputStageEvalBufferImpl<OutputStageSaturatingCastToInt16,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::int16_t, Size> OutputType;
  static_assert(InputType::kRegisterLanes == 1,
                "This path is only for scalar values");

  typedef OutputStageSaturatingCastToInt16 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      std::int32_t data = input.reg[i];
      output.reg[i] = data > 32767 ? 32767 : data < -32768 ? -32768 : data;
    }
    return output;
  }
};

// Implementation of OutputStageTruncatingCastToUint8 for scalar data
template <int Size>
struct OutputStageEvalBufferImpl<OutputStageTruncatingCastToUint8,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::uint8_t, Size> OutputType;
  static_assert(InputType::kRegisterLanes == 1,
                "This path is only for scalar values");

  typedef OutputStageTruncatingCastToUint8 OutputStage;

  OutputStageEvalBufferImpl(const OutputStage&) {}

  OutputType Eval(InputType input) const {
    OutputType output;
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      output.reg[i] = input.reg[i];
    }
    return output;
  }
};

template <int Rows, int Cols, typename VectorType>
struct OutputStageEvalImpl<OutputStageBiasAddition<VectorType>,
                           RegisterBlock<std::int32_t, Rows, Cols>> {
  typedef RegisterBlock<std::int32_t, Rows, Cols> InputType;
  typedef RegisterBlock<std::int32_t, Rows, Cols> OutputType;
  typedef OutputStageBiasAddition<VectorType> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const int pos = VectorType::kShape == VectorShape::Row ? col : row;
    return BroadcastAdd<InputType>(
        input, LoadForBroadcasting<InputType>(output_stage.bias_vector, pos));
  }

  const OutputStage& output_stage;
};

template <int Size>
struct OutputStageEvalBufferImpl<OutputStageClamp,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::int32_t, Size> OutputType;

  typedef OutputStageClamp OutputStage;

  OutputStageEvalBufferImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input) const {
    using RegisterType = typename InputType::RegisterType;
    const RegisterType min = Dup<RegisterType>(output_stage.min);
    const RegisterType max = Dup<RegisterType>(output_stage.max);
    OutputType output;
    for (int i = 0; i < InputType::kRegisterCount; i++) {
      output.reg[i] = Min(Max(input.reg[i], min), max);
    }
    return output;
  }

  const OutputStage& output_stage;
};

template <int Size>
struct OutputStageEvalBufferImpl<OutputStageTanh,
                                 RegisterBuffer<std::int32_t, Size>> {
  typedef RegisterBuffer<std::int32_t, Size> InputType;
  typedef RegisterBuffer<std::int32_t, Size> OutputType;
  using RegisterType = typename InputType::RegisterType;
  typedef RegisterType DataType;
  typedef OutputStageTanh OutputStage;

  OutputStageEvalBufferImpl(const OutputStage& s) : output_stage(s) {
    const std::int32_t real_zero_as_int32 = output_stage.real_zero_as_int32;
    const std::int32_t real_amplitude_as_int32 =
        output_stage.real_amplitude_as_int32;

    input_cutoff_min = real_zero_as_int32 - 8 * real_amplitude_as_int32;
    input_cutoff_max = real_zero_as_int32 + 8 * real_amplitude_as_int32;
    output_min = real_zero_as_int32 - real_amplitude_as_int32;
    output_max = real_zero_as_int32 + real_amplitude_as_int32;

    double inverse_amplitude_normalized_double = 1.0 / real_amplitude_as_int32;
    inverse_amplitude_neg_exponent = 0;
    while (inverse_amplitude_normalized_double < 0.5) {
      inverse_amplitude_normalized_double *= 2;
      inverse_amplitude_neg_exponent++;
    }
    inverse_amplitude_normalized = FixedPoint<DataType, 0>::FromDouble(
        inverse_amplitude_normalized_double);

    double amplitude_normalized_double = real_amplitude_as_int32;
    amplitude_exponent = 0;
    while (amplitude_normalized_double >= 1.0) {
      amplitude_normalized_double *= 0.5;
      amplitude_exponent++;
    }
    amplitude_normalized =
        FixedPoint<DataType, 0>::FromDouble(amplitude_normalized_double);
  }

  OutputType Eval(InputType input) const {
    const std::int32_t real_zero_as_int32 = output_stage.real_zero_as_int32;

    typedef FixedPoint<DataType, 3> F3;
    typedef FixedPoint<DataType, 0> F0;

    OutputType output;

    for (int i = 0; i < OutputType::kRegisterCount; i++) {
      // fixed-point affine transformation
      DataType input_centered =
          Sub(input.reg[i], Dup<DataType>(real_zero_as_int32));
      F3 fixedpoint_input =
          F3::FromRaw(input_centered) * inverse_amplitude_normalized;
      // left shift
      fixedpoint_input.raw() = ShiftLeft(fixedpoint_input.raw(),
                                         28 - inverse_amplitude_neg_exponent);
      // fixed-point tanh and multiplication
      F0 fixedpoint_output = tanh(fixedpoint_input) * amplitude_normalized;
      // right shift
      DataType int32_output =
          Add(Dup<DataType>(real_zero_as_int32),
              ShiftRight(fixedpoint_output.raw(), 31 - amplitude_exponent));

      DataType mask_if_below_cutoff_min =
          MaskIfLessThanOrEqual(input.reg[i], Dup<DataType>(input_cutoff_min));
      DataType mask_if_above_cutoff_max = MaskIfGreaterThanOrEqual(
          input.reg[i], Dup<DataType>(input_cutoff_max));

      output.reg[i] = SelectUsingMask(
          mask_if_below_cutoff_min, Dup<DataType>(output_min),
          SelectUsingMask(mask_if_above_cutoff_max, Dup<DataType>(output_max),
                          int32_output));
    }
    return output;
  }

  const OutputStage& output_stage;
  std::int32_t input_cutoff_min, input_cutoff_max;
  std::int32_t output_min, output_max;
  FixedPoint<DataType, 0> inverse_amplitude_normalized;
  int inverse_amplitude_neg_exponent;
  FixedPoint<DataType, 0> amplitude_normalized;
  int amplitude_exponent;
};

// OutputPipelineOutputType is a helper to determine the output data type of a
// pipeline, for a
// given input data type. It is a recursive template; see the explanation on
// OutputPipelineEvalImpl below.
template <typename OutputPipelineType, int FirstStage, typename InputType,
          bool StopRecursion =
              FirstStage == std::tuple_size<OutputPipelineType>::value>
struct OutputPipelineOutputType {
  typedef typename std::tuple_element<FirstStage, OutputPipelineType>::type
      FirstStageType;
  typedef typename OutputStageEvalImpl<FirstStageType, InputType>::OutputType
      FirstStageOutputType;
  typedef typename OutputPipelineOutputType<OutputPipelineType, FirstStage + 1,
                                            FirstStageOutputType>::Type Type;
};

template <typename OutputPipelineType, int FirstStage, typename InputType>
struct OutputPipelineOutputType<OutputPipelineType, FirstStage, InputType,
                                true> {
  typedef InputType Type;
};

// OutputPipelineEvalImpl is a helper to implement the evaluation of
// the whole pipeline. It is a recursive template to implement compile-time
// unrolling of the loop over all pipeline stages. The 'FirstStage' parameter
// is how we implement recursion: each specialization implements only
// evaluation starting at 'FirstStage'. The StopRecursion parameter is just a
// helper to implement the termination of the recursion as a partial
// specialization below.
template <typename OutputPipelineType, int FirstStage, typename InputType,
          bool StopRecursion =
              FirstStage == std::tuple_size<OutputPipelineType>::value>
struct OutputPipelineEvalImpl {
  typedef typename std::tuple_element<FirstStage, OutputPipelineType>::type
      FirstStageType;
  typedef typename OutputStageEvalImpl<FirstStageType, InputType>::OutputType
      FirstStageOutputType;
  typedef typename OutputPipelineOutputType<OutputPipelineType, FirstStage,
                                            InputType>::Type OutputType;

  OutputPipelineEvalImpl(const OutputPipelineType& output_pipeline)
      : head_impl(std::get<FirstStage>(output_pipeline)),
        tail_impl(output_pipeline) {}

  OutputType Eval(InputType input, int row, int col) const {
    // Evaluate the first stage.
    FirstStageOutputType first_stage_output = head_impl.Eval(input, row, col);
    // Recurse into the remaining stages.
    return tail_impl.Eval(first_stage_output, row, col);
  }

  const OutputStageEvalImpl<FirstStageType, InputType> head_impl;
  const OutputPipelineEvalImpl<OutputPipelineType, FirstStage + 1,
                               FirstStageOutputType>
      tail_impl;
};

// Specialization on 'StopRecursion' for terminating the recursion.
template <typename OutputPipelineType, int FirstStage, typename InputType>
struct OutputPipelineEvalImpl<OutputPipelineType, FirstStage, InputType, true> {
  OutputPipelineEvalImpl(const OutputPipelineType&) {}

  InputType Eval(InputType input, int, int) const {
    // Terminating the recursion.
    return input;
  }
};

template <typename RegisterBlockType, typename DstType>
struct StoreFinalOutputImpl {
  static_assert(std::is_same<RegisterBlockType, void>::value,
                "This generic impl should never be hit");
};

template <typename ScalarType, int Rows, int Cols, typename DstType>
struct StoreFinalOutputImpl<RegisterBlock<ScalarType, Rows, Cols>, DstType> {
  using RegisterBlockType = RegisterBlock<ScalarType, Rows, Cols>;
  static void Run(const RegisterBlockType& src, DstType* dst, int row,
                  int col) {
    for (int r = 0; r < Rows; r++) {
      for (int c = 0; c < Cols; c++) {
        *dst->data(row + r, col + c) = src.buf.reg[r + c * Rows];
      }
    }
  }
};

// StoreFinalOutput takes the final value at the end of the output pipeline and
// stores it into the destination matrix. It can be specialized for different
// data types; the generic implementation here is typically used only for plain
// old scalar (not SIMD) types.
template <typename RegisterBlockType, typename DstType>
void StoreFinalOutput(RegisterBlockType src, DstType* dst, int row, int col) {
  StoreFinalOutputImpl<RegisterBlockType, DstType>::Run(src, dst, row, col);
}

template <typename OutputPipelineType, typename InputType>
struct OutputPipelineExecutor {
  OutputPipelineExecutor(const OutputPipelineType& output_pipeline)
      : output_pipeline_eval_impl_(output_pipeline) {}

  // Execute is the entry point into the output pipeline evaluation
  // code. It should be the only thing that unpack code calls. It takes the
  // result
  // of the unpack stage and stores it into the destination matrix.
  template <typename DstType>
  void Execute(InputType input, DstType* dst, int src_global_row,
               int src_global_col, int dst_row, int dst_col) const {
    // Statically assert that the output pipeline matches the given destination
    // matrix's scalar type.
    typedef typename OutputPipelineOutputType<
        OutputPipelineType, 0, InputType>::Type::BufferType::ScalarType

        ScalarOutputType;
    typedef typename DstType::Scalar ScalarDstType;
    static_assert(std::is_same<ScalarOutputType, ScalarDstType>::value,
                  "mismatched destination scalar type and output pipeline");

    // Evaluate the output pipeline.
    auto output =
        output_pipeline_eval_impl_.Eval(input, src_global_row, src_global_col);
    // Store the result into the destination matrix.
    StoreFinalOutput(output, dst, dst_row, dst_col);
  }

  const OutputPipelineEvalImpl<OutputPipelineType, 0, InputType>
      output_pipeline_eval_impl_;
};

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "output_neon.h"
#elif defined(GEMMLOWP_SSE4)
#include "output_sse.h"
#elif defined(GEMMLOWP_MSA)
#include "output_msa.h"
#endif

#endif  // GEMMLOWP_INTERNAL_OUTPUT_H_
