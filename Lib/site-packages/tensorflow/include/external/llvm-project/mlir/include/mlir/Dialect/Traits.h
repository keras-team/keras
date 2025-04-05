//===- Traits.h - Common op traits shared by dialects -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common op traits that are not core to MLIR but can be
// shared by multiple dialects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRAITS_H
#define MLIR_DIALECT_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

// These functions are out-of-line implementations of the methods in the
// corresponding trait classes.  This avoids them being template
// instantiated/duplicated.
namespace impl {
LogicalResult verifyCompatibleOperandBroadcast(Operation *op);
} // namespace impl

namespace util {
/// Returns true and sets `resultShape` to the broadcasted shape from the two
/// given shapes if they are broadcast compatible. Returns false and clears
/// `resultShape` otherwise.
///
/// The rules for determining the result shape are:
///
/// Zip together the dimensions in the two given shapes by prepending the shape
/// with less dimensions with 1s. For each dimension pair, deduces the result
/// dimension according to the following order:
/// - If there are unknown dimensions, follows the TensorFlow behavior:
///   - If either dimension is greater than 1, we assume that the program is
///     correct, and the other dimension will be broadcast to match it.
///   - If either dimension is 1, the other dimension is the result.
///   - Otherwise, the result dimension is unknown dimension.
/// - If one of the dimension is 1, the other dimension is the result.
/// - If two dimensions are the same, that's the result.
/// - Otherwise, incompatible shape.
bool getBroadcastedShape(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                         SmallVectorImpl<int64_t> &resultShape);

/// Returns true if a broadcast between n shapes is guaranteed to be
/// successful and not result in an error. False does not guarantee that the
/// shapes are not broadcastable; it might guarantee that they are not
/// broadcastable or it might mean that this function does not have enough
/// information to know.
///
/// Conceptually, this returns true if getBroadcastedShape would have returned
/// true and vice versa, with one exception. If a dimension is unknown in both
/// shapes, getBroadcastedShape would return true and have a result with unknown
/// dimension, while this function will return false because it's possible for
/// both shapes to have a dimension greater than 1 and different which would
/// fail to broadcast.
bool staticallyKnownBroadcastable(ArrayRef<SmallVector<int64_t, 6>> shapes);
bool staticallyKnownBroadcastable(ArrayRef<int64_t> shape1,
                                  ArrayRef<int64_t> shape2);

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returned type may have dynamic shape if
/// either of the input types has dynamic shape. Returns null type if the two
/// given types are not broadcast-compatible.
///
/// elementType, if specified, will be used as the element type of the
/// broadcasted result type. Otherwise it is required that the element type of
/// type1 and type2 is the same and this element type will be used as the
/// resultant element type.
Type getBroadcastedType(Type type1, Type type2, Type elementType = nullptr);

} // namespace util

/// Trait for ops that are known to have broadcast compatible operands and
/// result types. Specifically, starting from the most varying dimension, each
/// dimension pair of the operands' shapes should either be the same or one
/// of them is one. Also, the results's shapes should have the corresponding
/// dimension equal to the larger one, if known. Shapes are checked partially if
/// ranks or dimensions are not known. For example, an op with tensor<?x2xf32>
/// and tensor<2xf32> as operand types and tensor<5x3x2xi16> as the result
/// type has broadcast compatible operands ns result types.
template <typename ConcreteType>
class ResultsBroadcastableShape
    : public TraitBase<ConcreteType, ResultsBroadcastableShape> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    return impl::verifyCompatibleOperandBroadcast(op);
  }
};

} // namespace OpTrait
} // namespace mlir

#endif // MLIR_DIALECT_TRAITS_H
