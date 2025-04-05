//===- Utils.h - General Arith transformation utilities ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the Arith dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_UTILS_UTILS_H
#define MLIR_DIALECT_ARITH_UTILS_UTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

using ReassociationIndices = SmallVector<int64_t, 2>;

/// Infer the output shape for a {memref|tensor}.expand_shape when it is
/// possible to do so.
///
/// Note: This should *only* be used to implement
/// `ExpandShapeOp::inferOutputShape` in both the memref and tensor namespaces.
/// If you need to infer the output shape you should use the static method of
/// `ExpandShapeOp` instead of calling this.
///
/// `inputShape` is the shape of the tensor or memref being expanded as a
/// sequence of SSA values or constants. `expandedType` is the output shape of
/// the expand_shape operation. `reassociation` is the reassociation denoting
/// the output dims each input dim is mapped to.
///
/// Returns the output shape in `outputShape` and `staticOutputShape`, following
/// the conventions for the output_shape and static_output_shape inputs to the
/// expand_shape ops.
std::optional<SmallVector<OpFoldResult>>
inferExpandShapeOutputShape(OpBuilder &b, Location loc, ShapedType expandedType,
                            ArrayRef<ReassociationIndices> reassociation,
                            ArrayRef<OpFoldResult> inputShape);

/// Matches a ConstantIndexOp.
detail::op_matcher<arith::ConstantIndexOp> matchConstantIndex();

llvm::SmallBitVector getPositionsOfShapeOne(unsigned rank,
                                            ArrayRef<int64_t> shape);

/// Converts an OpFoldResult to a Value. Returns the fold result if it casts to
/// a Value or creates a ConstantOp if it casts to an Integer Attribute.
/// Other attribute types are not supported.
Value getValueOrCreateConstantIntOp(OpBuilder &b, Location loc,
                                    OpFoldResult ofr);

/// Converts an OpFoldResult to a Value. Returns the fold result if it casts to
/// a Value or creates a ConstantIndexOp if it casts to an Integer Attribute.
/// Other attribute types are not supported.
Value getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                      OpFoldResult ofr);

/// Similar to the other overload, but converts multiple OpFoldResults into
/// Values.
SmallVector<Value>
getValueOrCreateConstantIndexOp(OpBuilder &b, Location loc,
                                ArrayRef<OpFoldResult> valueOrAttrVec);

/// Create a cast from an index-like value (index or integer) to another
/// index-like value. If the value type and the target type are the same, it
/// returns the original value.
Value getValueOrCreateCastToIndexLike(OpBuilder &b, Location loc,
                                      Type targetType, Value value);

/// Converts a scalar value `operand` to type `toType`. If the value doesn't
/// convert, a warning will be issued and the operand is returned as is (which
/// will presumably yield a verification issue downstream).
Value convertScalarToDtype(OpBuilder &b, Location loc, Value operand,
                           Type toType, bool isUnsignedCast);

/// Create a constant of type `type` at location `loc` whose value is `value`
/// (an APInt or APFloat whose type must match the element type of `type`).
/// If `type` is a shaped type, create a splat constant of the given value.
/// Constants are folded if possible.
Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  const APInt &value);
Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  int64_t value);
Value createScalarOrSplatConstant(OpBuilder &builder, Location loc, Type type,
                                  const APFloat &value);

/// Returns the int type of the integer in ofr.
/// Other attribute types are not supported.
Type getType(OpFoldResult ofr);

/// Helper struct to build simple arithmetic quantities with minimal type
/// inference support.
struct ArithBuilder {
  ArithBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value _and(Value lhs, Value rhs);
  Value add(Value lhs, Value rhs);
  Value sub(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);

private:
  OpBuilder &b;
  Location loc;
};

namespace arith {

// Build the product of a sequence.
// If values = (v0, v1, ..., vn) than the returned
// value is v0 * v1 * ... * vn.
// All values must have the same type.
//
// The version without `resultType` must contain at least one element in values.
// Then the result will have the same type as the elements in `values`.
// If `values` is empty in the version with `resultType` returns 1 with type
// `resultType`.
Value createProduct(OpBuilder &builder, Location loc, ArrayRef<Value> values);
Value createProduct(OpBuilder &builder, Location loc, ArrayRef<Value> values,
                    Type resultType);
} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_UTILS_UTILS_H
