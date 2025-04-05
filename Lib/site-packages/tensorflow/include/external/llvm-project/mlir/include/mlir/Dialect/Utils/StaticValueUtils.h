//===- StaticValueUtils.h - Utilities for static values ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for dealing with static values, e.g.,
// converting back and forth between Value and OpFoldResult. Such functionality
// is used in multiple dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UTILS_STATICVALUEUTILS_H
#define MLIR_DIALECT_UTILS_STATICVALUEUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace mlir {

/// Return true if `v` is an IntegerAttr with value `0` of a ConstantIndexOp
/// with attribute with value `0`.
bool isZeroIndex(OpFoldResult v);

/// Represents a range (offset, size, and stride) where each element of the
/// triple may be dynamic or static.
struct Range {
  OpFoldResult offset;
  OpFoldResult size;
  OpFoldResult stride;
};

/// Given an array of Range values, return a tuple of (offset vector, sizes
/// vector, and strides vector) formed by separating out the individual
/// elements of each range.
std::tuple<SmallVector<OpFoldResult>, SmallVector<OpFoldResult>,
           SmallVector<OpFoldResult>>
getOffsetsSizesAndStrides(ArrayRef<Range> ranges);

/// Helper function to dispatch an OpFoldResult into `staticVec` if:
///   a) it is an IntegerAttr
/// In other cases, the OpFoldResult is dispached to the `dynamicVec`.
/// In such dynamic cases, ShapedType::kDynamic is also pushed to
/// `staticVec`. This is useful to extract mixed static and dynamic entries
/// that come from an AttrSizedOperandSegments trait.
void dispatchIndexOpFoldResult(OpFoldResult ofr,
                               SmallVectorImpl<Value> &dynamicVec,
                               SmallVectorImpl<int64_t> &staticVec);

/// Helper function to dispatch multiple OpFoldResults according to the
/// behavior of `dispatchIndexOpFoldResult(OpFoldResult ofr` for a single
/// OpFoldResult.
void dispatchIndexOpFoldResults(ArrayRef<OpFoldResult> ofrs,
                                SmallVectorImpl<Value> &dynamicVec,
                                SmallVectorImpl<int64_t> &staticVec);

/// Extract integer values from the assumed ArrayAttr of IntegerAttr.
template <typename IntTy>
SmallVector<IntTy> extractFromIntegerArrayAttr(Attribute attr) {
  return llvm::to_vector(
      llvm::map_range(cast<ArrayAttr>(attr), [](Attribute a) -> IntTy {
        return cast<IntegerAttr>(a).getInt();
      }));
}

/// Given a value, try to extract a constant Attribute. If this fails, return
/// the original value.
OpFoldResult getAsOpFoldResult(Value val);
/// Given an array of values, try to extract a constant Attribute from each
/// value. If this fails, return the original value.
SmallVector<OpFoldResult> getAsOpFoldResult(ValueRange values);
/// Convert `arrayAttr` to a vector of OpFoldResult.
SmallVector<OpFoldResult> getAsOpFoldResult(ArrayAttr arrayAttr);

/// Convert int64_t to integer attributes of index type and return them as
/// OpFoldResult.
OpFoldResult getAsIndexOpFoldResult(MLIRContext *ctx, int64_t val);
SmallVector<OpFoldResult> getAsIndexOpFoldResult(MLIRContext *ctx,
                                                 ArrayRef<int64_t> values);

/// If ofr is a constant integer or an IntegerAttr, return the integer.
std::optional<int64_t> getConstantIntValue(OpFoldResult ofr);
/// If all ofrs are constant integers or IntegerAttrs, return the integers.
std::optional<SmallVector<int64_t>>
getConstantIntValues(ArrayRef<OpFoldResult> ofrs);

/// Return true if `ofr` is constant integer equal to `value`.
bool isConstantIntValue(OpFoldResult ofr, int64_t value);

/// Return true if ofr1 and ofr2 are the same integer constant attribute
/// values or the same SSA value. Ignore integer bitwitdh and type mismatch
/// that come from the fact there is no IndexAttr and that IndexType have no
/// bitwidth.
bool isEqualConstantIntOrValue(OpFoldResult ofr1, OpFoldResult ofr2);
bool isEqualConstantIntOrValueArray(ArrayRef<OpFoldResult> ofrs1,
                                    ArrayRef<OpFoldResult> ofrs2);

// To convert an OpFoldResult to a Value of index type, see:
//   mlir/include/mlir/Dialect/Arith/Utils/Utils.h
// TODO: find a better common landing place.
//
// Value getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
//                                       OpFoldResult ofr);

// To convert an OpFoldResult to a Value of index type, see:
//   mlir/include/mlir/Dialect/Arith/Utils/Utils.h
// TODO: find a better common landing place.
//
// SmallVector<Value>
// getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
//                                 ArrayRef<OpFoldResult> valueOrAttrVec);

/// Return a vector of OpFoldResults with the same size a staticValues, but
/// all elements for which ShapedType::isDynamic is true, will be replaced by
/// dynamicValues.
SmallVector<OpFoldResult> getMixedValues(ArrayRef<int64_t> staticValues,
                                         ValueRange dynamicValues, Builder &b);

/// Decompose a vector of mixed static or dynamic values into the
/// corresponding pair of arrays. This is the inverse function of
/// `getMixedValues`.
std::pair<SmallVector<int64_t>, SmallVector<Value>>
decomposeMixedValues(const SmallVectorImpl<OpFoldResult> &mixedValues);

/// Helper to sort `values` according to matching `keys`.
SmallVector<Value>
getValuesSortedByKey(ArrayRef<Attribute> keys, ArrayRef<Value> values,
                     llvm::function_ref<bool(Attribute, Attribute)> compare);
SmallVector<OpFoldResult>
getValuesSortedByKey(ArrayRef<Attribute> keys, ArrayRef<OpFoldResult> values,
                     llvm::function_ref<bool(Attribute, Attribute)> compare);
SmallVector<int64_t>
getValuesSortedByKey(ArrayRef<Attribute> keys, ArrayRef<int64_t> values,
                     llvm::function_ref<bool(Attribute, Attribute)> compare);

/// Helper function to check whether the passed in `sizes` or `offsets` are
/// valid. This can be used to re-check whether dimensions are still valid
/// after constant folding the dynamic dimensions.
bool hasValidSizesOffsets(SmallVector<int64_t> sizesOrOffsets);

/// Helper function to check whether the passed in `strides` are valid. This
/// can be used to re-check whether dimensions are still valid after constant
/// folding the dynamic dimensions.
bool hasValidStrides(SmallVector<int64_t> strides);

/// Returns "success" when any of the elements in `ofrs` is a constant value. In
/// that case the value is replaced by an attribute. Returns "failure" when no
/// folding happened. If `onlyNonNegative` and `onlyNonZero` are set, only
/// non-negative and non-zero constant values are folded respectively.
LogicalResult foldDynamicIndexList(SmallVectorImpl<OpFoldResult> &ofrs,
                                   bool onlyNonNegative = false,
                                   bool onlyNonZero = false);

/// Returns "success" when any of the elements in `offsetsOrSizes` is a
/// constant value. In that case the value is replaced by an attribute. Returns
/// "failure" when no folding happened. Invalid values are not folded to avoid
/// canonicalization crashes.
LogicalResult
foldDynamicOffsetSizeList(SmallVectorImpl<OpFoldResult> &offsetsOrSizes);

/// Returns "success" when any of the elements in `strides` is a constant
/// value. In that case the value is replaced by an attribute. Returns
/// "failure" when no folding happened. Invalid values are not folded to avoid
/// canonicalization crashes.
LogicalResult foldDynamicStrideList(SmallVectorImpl<OpFoldResult> &strides);

/// Return the number of iterations for a loop with a lower bound `lb`, upper
/// bound `ub` and step `step`.
std::optional<int64_t> constantTripCount(OpFoldResult lb, OpFoldResult ub,
                                         OpFoldResult step);

/// Idiomatic saturated operations on values like offsets, sizes, and strides.
struct SaturatedInteger {
  static SaturatedInteger wrap(int64_t v) {
    return (ShapedType::isDynamic(v)) ? SaturatedInteger{true, 0}
                                      : SaturatedInteger{false, v};
  }
  int64_t asInteger() { return saturated ? ShapedType::kDynamic : v; }
  FailureOr<SaturatedInteger> desaturate(SaturatedInteger other) {
    if (saturated && !other.saturated)
      return other;
    if (!saturated && !other.saturated && v != other.v)
      return failure();
    return *this;
  }
  bool operator==(SaturatedInteger other) {
    return (saturated && other.saturated) ||
           (!saturated && !other.saturated && v == other.v);
  }
  bool operator!=(SaturatedInteger other) { return !(*this == other); }
  SaturatedInteger operator+(SaturatedInteger other) {
    if (saturated || other.saturated)
      return SaturatedInteger{true, 0};
    return SaturatedInteger{false, other.v + v};
  }
  SaturatedInteger operator*(SaturatedInteger other) {
    // Multiplication with 0 is always 0.
    if (!other.saturated && other.v == 0)
      return SaturatedInteger{false, 0};
    if (!saturated && v == 0)
      return SaturatedInteger{false, 0};
    // Otherwise, if this or the other integer is dynamic, so is the result.
    if (saturated || other.saturated)
      return SaturatedInteger{true, 0};
    return SaturatedInteger{false, other.v * v};
  }
  bool saturated = true;
  int64_t v = 0;
};

} // namespace mlir

#endif // MLIR_DIALECT_UTILS_STATICVALUEUTILS_H
