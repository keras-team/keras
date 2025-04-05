//===- Arith.h - Arith dialect ------------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_IR_ARITH_H_
#define MLIR_DIALECT_ARITH_IR_ARITH_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "llvm/ADT/StringExtras.h"

//===----------------------------------------------------------------------===//
// ArithDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ArithOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Arith Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/ArithOpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Arith/IR/ArithOpsAttributes.h.inc"

//===----------------------------------------------------------------------===//
// Arith Interfaces
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// Arith Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Arith/IR/ArithOps.h.inc"

namespace mlir {
namespace arith {

/// Specialization of `arith.constant` op that returns an integer value.
class ConstantIntOp : public arith::ConstantOp {
public:
  using arith::ConstantOp::ConstantOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ConstantOp>(); }

  /// Build a constant int op that produces an integer of the specified width.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    unsigned width);

  /// Build a constant int op that produces an integer of the specified type,
  /// which must be an integer type.
  static void build(OpBuilder &builder, OperationState &result, int64_t value,
                    Type type);

  inline int64_t value() {
    return cast<IntegerAttr>(arith::ConstantOp::getValue()).getInt();
  }

  static bool classof(Operation *op);
};

/// Specialization of `arith.constant` op that returns a floating point value.
class ConstantFloatOp : public arith::ConstantOp {
public:
  using arith::ConstantOp::ConstantOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ConstantOp>(); }

  /// Build a constant float op that produces a float of the specified type.
  static void build(OpBuilder &builder, OperationState &result,
                    const APFloat &value, FloatType type);

  inline APFloat value() {
    return cast<FloatAttr>(arith::ConstantOp::getValue()).getValue();
  }

  static bool classof(Operation *op);
};

/// Specialization of `arith.constant` op that returns an integer of index type.
class ConstantIndexOp : public arith::ConstantOp {
public:
  using arith::ConstantOp::ConstantOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ConstantOp>(); }
  /// Build a constant int op that produces an index.
  static void build(OpBuilder &builder, OperationState &result, int64_t value);

  inline int64_t value() {
    return cast<IntegerAttr>(arith::ConstantOp::getValue()).getInt();
  }

  static bool classof(Operation *op);
};

} // namespace arith
} // namespace mlir

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace arith {

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known integer
/// comparison predicates.
bool applyCmpPredicate(arith::CmpIPredicate predicate, const APInt &lhs,
                       const APInt &rhs);

/// Compute `lhs` `pred` `rhs`, where `pred` is one of the known floating point
/// comparison predicates.
bool applyCmpPredicate(arith::CmpFPredicate predicate, const APFloat &lhs,
                       const APFloat &rhs);

/// Returns the identity value attribute associated with an AtomicRMWKind op.
/// `useOnlyFiniteValue` defines whether the identity value should steer away
/// from infinity representations or anything that is not a proper finite
/// number.
/// E.g., The identity value for maxf is in theory `-Inf`, but if we want to
/// stay in the finite range, it would be `BiggestRepresentableNegativeFloat`.
/// The purpose of this boolean is to offer constants that will play nice
/// with fast math related optimizations.
TypedAttr getIdentityValueAttr(AtomicRMWKind kind, Type resultType,
                               OpBuilder &builder, Location loc,
                               bool useOnlyFiniteValue = false);

/// Return the identity numeric value associated to the give op. Return
/// std::nullopt if there is no known neutral element.
/// If `op` has `FastMathFlags::ninf`, only finite values will be used
/// as neutral element.
std::optional<TypedAttr> getNeutralElement(Operation *op);

/// Returns the identity value associated with an AtomicRMWKind op.
/// \see getIdentityValueAttr for a description of what `useOnlyFiniteValue`
/// does.
Value getIdentityValue(AtomicRMWKind op, Type resultType, OpBuilder &builder,
                       Location loc, bool useOnlyFiniteValue = false);

/// Returns the value obtained by applying the reduction operation kind
/// associated with a binary AtomicRMWKind op to `lhs` and `rhs`.
Value getReductionOp(AtomicRMWKind op, OpBuilder &builder, Location loc,
                     Value lhs, Value rhs);

arith::CmpIPredicate invertPredicate(arith::CmpIPredicate pred);
} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_IR_ARITH_H_
