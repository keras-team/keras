//===- InferIntRangeCommon.cpp - Inference for common ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares implementations of range inference for operations that are
// common to both the `arith` and `index` dialects to facilitate reuse.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_UTILS_INFERINTRANGECOMMON_H
#define MLIR_INTERFACES_UTILS_INFERINTRANGECOMMON_H

#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include <optional>

namespace mlir {
namespace intrange {
/// Function that performs inference on an array of `ConstantIntRanges`,
/// abstracted away here to permit writing the function that handles both
/// 64- and 32-bit index types.
using InferRangeFn =
    std::function<ConstantIntRanges(ArrayRef<ConstantIntRanges>)>;

/// Function that performs inferrence on an array of `IntegerValueRange`.
using InferIntegerValueRangeFn =
    std::function<IntegerValueRange(ArrayRef<IntegerValueRange>)>;

static constexpr unsigned indexMinWidth = 32;
static constexpr unsigned indexMaxWidth = 64;

enum class CmpMode : uint32_t { Both, Signed, Unsigned };

enum class OverflowFlags : uint32_t {
  None = 0,
  Nsw = 1,
  Nuw = 2,
  LLVM_MARK_AS_BITMASK_ENUM(Nuw)
};

/// Function that performs inference on an array of `ConstantIntRanges` while
/// taking special overflow behavior into account.
using InferRangeWithOvfFlagsFn =
    function_ref<ConstantIntRanges(ArrayRef<ConstantIntRanges>, OverflowFlags)>;

/// Compute `inferFn` on `ranges`, whose size should be the index storage
/// bitwidth. Then, compute the function on `argRanges` again after truncating
/// the ranges to 32 bits. Finally, if the truncation of the 64-bit result is
/// equal to the 32-bit result, use it (to preserve compatibility with folders
/// and inference precision), and take the union of the results otherwise.
///
/// The `mode` argument specifies if the unsigned, signed, or both results of
/// the inference computation should be used when comparing the results.
ConstantIntRanges inferIndexOp(const InferRangeFn &inferFn,
                               ArrayRef<ConstantIntRanges> argRanges,
                               CmpMode mode);

/// Independently zero-extend the unsigned values and sign-extend the signed
/// values in `range` to `destWidth` bits, returning the resulting range.
ConstantIntRanges extRange(const ConstantIntRanges &range, unsigned destWidth);

/// Use the unsigned values in `range` to zero-extend it to `destWidth`.
ConstantIntRanges extUIRange(const ConstantIntRanges &range,
                             unsigned destWidth);

/// Use the signed values in `range` to sign-extend it to `destWidth`.
ConstantIntRanges extSIRange(const ConstantIntRanges &range,
                             unsigned destWidth);

/// Truncate `range` to `destWidth` bits, taking care to handle cases such as
/// the truncation of [255, 256] to i8 not being a uniform range.
ConstantIntRanges truncRange(const ConstantIntRanges &range,
                             unsigned destWidth);

ConstantIntRanges inferAdd(ArrayRef<ConstantIntRanges> argRanges,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferSub(ArrayRef<ConstantIntRanges> argRanges,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferMul(ArrayRef<ConstantIntRanges> argRanges,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferDivS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferDivU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferCeilDivS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferCeilDivU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferFloorDivS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferRemS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferRemU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMaxS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMaxU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMinS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferMinU(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferAnd(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferOr(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferXor(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferShl(ArrayRef<ConstantIntRanges> argRanges,
                           OverflowFlags ovfFlags = OverflowFlags::None);

ConstantIntRanges inferShrS(ArrayRef<ConstantIntRanges> argRanges);

ConstantIntRanges inferShrU(ArrayRef<ConstantIntRanges> argRanges);

/// Copy of the enum from `arith` and `index` to allow the common integer range
/// infrastructure to not depend on either dialect.
enum class CmpPredicate : uint64_t {
  eq,
  ne,
  slt,
  sle,
  sgt,
  sge,
  ult,
  ule,
  ugt,
  uge,
};

/// Returns a boolean value if `pred` is statically true or false for
/// anypossible inputs falling within `lhs` and `rhs`, and std::nullopt if the
/// value of the predicate cannot be determined.
std::optional<bool> evaluatePred(CmpPredicate pred,
                                 const ConstantIntRanges &lhs,
                                 const ConstantIntRanges &rhs);

} // namespace intrange
} // namespace mlir

#endif // MLIR_INTERFACES_UTILS_INFERINTRANGECOMMON_H
