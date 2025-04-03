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

// bit_depth.h: defines the settins controlling LHS/RHS bit depth

#ifndef GEMMLOWP_PUBLIC_BIT_DEPTH_H_
#define GEMMLOWP_PUBLIC_BIT_DEPTH_H_

namespace gemmlowp {

// The range of allowed values for an operand.
template <int tMinValue, int tMaxValue>
struct OperandRange {
  static constexpr int kMinValue = tMinValue;
  static constexpr int kMaxValue = tMaxValue;
  static_assert(kMinValue < kMaxValue, "");
};

using Uint8Range = OperandRange<0, 255>;
using Uint8RangeExcludingZero = OperandRange<1, 255>;

using Int8Range = OperandRange<-128, 127>;
using Int8RangeExcludingLow = OperandRange<-127, 127>;

template <typename tLhsRange, typename tRhsRange>
struct BitDepthParams {
  using LhsRange = tLhsRange;
  using RhsRange = tRhsRange;
};

// Default: LHS and RHS are 8bit.
using DefaultL8R8BitDepthParams = BitDepthParams<Uint8Range, Uint8Range>;

// Variant: LHS may not take the value 0. This allows using
// faster kernels using signed arithmetic, see
// NEON_64bit_GEMM_Int8Operands_Int32Accumulators_AccumTwoWithin16Bits
using L8R8WithLhsNonzeroBitDepthParams =
    BitDepthParams<Uint8RangeExcludingZero, Uint8Range>;

// Signed Variant: This allows using faster kernels using signed arithmetic, see
// NEON_64bit_GEMM_Int8Operands_Int32Accumulators_AccumTwoWithin16Bits
using SignedL8R8WithLhsNonzeroBitDepthParams =
    BitDepthParams<Int8RangeExcludingLow, Int8Range>;

// Deprecated: when gemmlowp used to allow requantizing 8bit
// inputs to less-than-8-bit depths, the public setting allowing
// that was DefaultL7R5BitDepthParams. That requantization
// feature has been removed, but as the whole point of that
// requantization was to make less-than-8-bit an internal
// optimization without any impact on the API (other than lowering
// accuracy), we can temporarily support users who were using it
// by mapping it to the default 8bit behavior.
using DefaultL7R5BitDepthParams = DefaultL8R8BitDepthParams;

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_BIT_DEPTH_H_
