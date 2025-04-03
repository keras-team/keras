//===- CommonFolders.h - Common Operation Folders----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares various common operation folders. These folders
// are intended to be used by dialects to support common folding behavior
// without requiring each dialect to provide its own implementation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_COMMONFOLDERS_H
#define MLIR_DIALECT_COMMONFOLDERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace mlir {
namespace ub {
class PoisonAttr;
}
/// Performs constant folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
/// Uses `resultType` for the type of the returned attribute.
/// Optional PoisonAttr template argument allows to specify 'poison' attribute
/// which will be directly propagated to result.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class PoisonAttr = ub::PoisonAttr,
          class CalculationT = function_ref<
              std::optional<ElementValueT>(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOpConditional(ArrayRef<Attribute> operands,
                                       Type resultType,
                                       CalculationT &&calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  static_assert(
      std::is_void_v<PoisonAttr> || !llvm::is_incomplete_v<PoisonAttr>,
      "PoisonAttr is undefined, either add a dependency on UB dialect or pass "
      "void as template argument to opt-out from poison semantics.");
  if constexpr (!std::is_void_v<PoisonAttr>) {
    if (isa_and_nonnull<PoisonAttr>(operands[0]))
      return operands[0];

    if (isa_and_nonnull<PoisonAttr>(operands[1]))
      return operands[1];
  }

  if (!resultType || !operands[0] || !operands[1])
    return {};

  if (isa<AttrElementT>(operands[0]) && isa<AttrElementT>(operands[1])) {
    auto lhs = cast<AttrElementT>(operands[0]);
    auto rhs = cast<AttrElementT>(operands[1]);
    if (lhs.getType() != rhs.getType())
      return {};

    auto calRes = calculate(lhs.getValue(), rhs.getValue());

    if (!calRes)
      return {};

    return AttrElementT::get(resultType, *calRes);
  }

  if (isa<SplatElementsAttr>(operands[0]) &&
      isa<SplatElementsAttr>(operands[1])) {
    // Both operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto lhs = cast<SplatElementsAttr>(operands[0]);
    auto rhs = cast<SplatElementsAttr>(operands[1]);
    if (lhs.getType() != rhs.getType())
      return {};

    auto elementResult = calculate(lhs.getSplatValue<ElementValueT>(),
                                   rhs.getSplatValue<ElementValueT>());
    if (!elementResult)
      return {};

    return DenseElementsAttr::get(cast<ShapedType>(resultType), *elementResult);
  }

  if (isa<ElementsAttr>(operands[0]) && isa<ElementsAttr>(operands[1])) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto lhs = cast<ElementsAttr>(operands[0]);
    auto rhs = cast<ElementsAttr>(operands[1]);
    if (lhs.getType() != rhs.getType())
      return {};

    auto maybeLhsIt = lhs.try_value_begin<ElementValueT>();
    auto maybeRhsIt = rhs.try_value_begin<ElementValueT>();
    if (!maybeLhsIt || !maybeRhsIt)
      return {};
    auto lhsIt = *maybeLhsIt;
    auto rhsIt = *maybeRhsIt;
    SmallVector<ElementValueT, 4> elementResults;
    elementResults.reserve(lhs.getNumElements());
    for (size_t i = 0, e = lhs.getNumElements(); i < e; ++i, ++lhsIt, ++rhsIt) {
      auto elementResult = calculate(*lhsIt, *rhsIt);
      if (!elementResult)
        return {};
      elementResults.push_back(*elementResult);
    }

    return DenseElementsAttr::get(cast<ShapedType>(resultType), elementResults);
  }
  return {};
}

/// Performs constant folding `calculate` with element-wise behavior on the two
/// attributes in `operands` and returns the result if possible.
/// Uses the operand element type for the element type of the returned
/// attribute.
/// Optional PoisonAttr template argument allows to specify 'poison' attribute
/// which will be directly propagated to result.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class PoisonAttr = ub::PoisonAttr,
          class CalculationT = function_ref<
              std::optional<ElementValueT>(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOpConditional(ArrayRef<Attribute> operands,
                                       CalculationT &&calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  static_assert(
      std::is_void_v<PoisonAttr> || !llvm::is_incomplete_v<PoisonAttr>,
      "PoisonAttr is undefined, either add a dependency on UB dialect or pass "
      "void as template argument to opt-out from poison semantics.");
  if constexpr (!std::is_void_v<PoisonAttr>) {
    if (isa_and_nonnull<PoisonAttr>(operands[0]))
      return operands[0];

    if (isa_and_nonnull<PoisonAttr>(operands[1]))
      return operands[1];
  }

  auto getResultType = [](Attribute attr) -> Type {
    if (auto typed = dyn_cast_or_null<TypedAttr>(attr))
      return typed.getType();
    return {};
  };

  Type lhsType = getResultType(operands[0]);
  Type rhsType = getResultType(operands[1]);
  if (!lhsType || !rhsType)
    return {};
  if (lhsType != rhsType)
    return {};

  return constFoldBinaryOpConditional<AttrElementT, ElementValueT, PoisonAttr,
                                      CalculationT>(
      operands, lhsType, std::forward<CalculationT>(calculate));
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class PoisonAttr = void,
          class CalculationT =
              function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOp(ArrayRef<Attribute> operands, Type resultType,
                            CalculationT &&calculate) {
  return constFoldBinaryOpConditional<AttrElementT, ElementValueT, PoisonAttr>(
      operands, resultType,
      [&](ElementValueT a, ElementValueT b) -> std::optional<ElementValueT> {
        return calculate(a, b);
      });
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class PoisonAttr = ub::PoisonAttr,
          class CalculationT =
              function_ref<ElementValueT(ElementValueT, ElementValueT)>>
Attribute constFoldBinaryOp(ArrayRef<Attribute> operands,
                            CalculationT &&calculate) {
  return constFoldBinaryOpConditional<AttrElementT, ElementValueT, PoisonAttr>(
      operands,
      [&](ElementValueT a, ElementValueT b) -> std::optional<ElementValueT> {
        return calculate(a, b);
      });
}

/// Performs constant folding `calculate` with element-wise behavior on the one
/// attributes in `operands` and returns the result if possible.
/// Optional PoisonAttr template argument allows to specify 'poison' attribute
/// which will be directly propagated to result.
template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class PoisonAttr = ub::PoisonAttr,
          class CalculationT =
              function_ref<std::optional<ElementValueT>(ElementValueT)>>
Attribute constFoldUnaryOpConditional(ArrayRef<Attribute> operands,
                                      CalculationT &&calculate) {
  assert(operands.size() == 1 && "unary op takes one operands");
  if (!operands[0])
    return {};

  static_assert(
      std::is_void_v<PoisonAttr> || !llvm::is_incomplete_v<PoisonAttr>,
      "PoisonAttr is undefined, either add a dependency on UB dialect or pass "
      "void as template argument to opt-out from poison semantics.");
  if constexpr (!std::is_void_v<PoisonAttr>) {
    if (isa<PoisonAttr>(operands[0]))
      return operands[0];
  }

  if (isa<AttrElementT>(operands[0])) {
    auto op = cast<AttrElementT>(operands[0]);

    auto res = calculate(op.getValue());
    if (!res)
      return {};
    return AttrElementT::get(op.getType(), *res);
  }
  if (isa<SplatElementsAttr>(operands[0])) {
    // Both operands are splats so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto op = cast<SplatElementsAttr>(operands[0]);

    auto elementResult = calculate(op.getSplatValue<ElementValueT>());
    if (!elementResult)
      return {};
    return DenseElementsAttr::get(op.getType(), *elementResult);
  } else if (isa<ElementsAttr>(operands[0])) {
    // Operands are ElementsAttr-derived; perform an element-wise fold by
    // expanding the values.
    auto op = cast<ElementsAttr>(operands[0]);

    auto maybeOpIt = op.try_value_begin<ElementValueT>();
    if (!maybeOpIt)
      return {};
    auto opIt = *maybeOpIt;
    SmallVector<ElementValueT> elementResults;
    elementResults.reserve(op.getNumElements());
    for (size_t i = 0, e = op.getNumElements(); i < e; ++i, ++opIt) {
      auto elementResult = calculate(*opIt);
      if (!elementResult)
        return {};
      elementResults.push_back(*elementResult);
    }
    return DenseElementsAttr::get(op.getShapedType(), elementResults);
  }
  return {};
}

template <class AttrElementT,
          class ElementValueT = typename AttrElementT::ValueType,
          class PoisonAttr = ub::PoisonAttr,
          class CalculationT = function_ref<ElementValueT(ElementValueT)>>
Attribute constFoldUnaryOp(ArrayRef<Attribute> operands,
                           CalculationT &&calculate) {
  return constFoldUnaryOpConditional<AttrElementT, ElementValueT, PoisonAttr>(
      operands, [&](ElementValueT a) -> std::optional<ElementValueT> {
        return calculate(a);
      });
}

template <
    class AttrElementT, class TargetAttrElementT,
    class ElementValueT = typename AttrElementT::ValueType,
    class TargetElementValueT = typename TargetAttrElementT::ValueType,
    class PoisonAttr = ub::PoisonAttr,
    class CalculationT = function_ref<TargetElementValueT(ElementValueT, bool)>>
Attribute constFoldCastOp(ArrayRef<Attribute> operands, Type resType,
                          CalculationT &&calculate) {
  assert(operands.size() == 1 && "Cast op takes one operand");
  if (!operands[0])
    return {};

  static_assert(
      std::is_void_v<PoisonAttr> || !llvm::is_incomplete_v<PoisonAttr>,
      "PoisonAttr is undefined, either add a dependency on UB dialect or pass "
      "void as template argument to opt-out from poison semantics.");
  if constexpr (!std::is_void_v<PoisonAttr>) {
    if (isa<PoisonAttr>(operands[0]))
      return operands[0];
  }

  if (isa<AttrElementT>(operands[0])) {
    auto op = cast<AttrElementT>(operands[0]);
    bool castStatus = true;
    auto res = calculate(op.getValue(), castStatus);
    if (!castStatus)
      return {};
    return TargetAttrElementT::get(resType, res);
  }
  if (isa<SplatElementsAttr>(operands[0])) {
    // The operand is a splat so we can avoid expanding the values out and
    // just fold based on the splat value.
    auto op = cast<SplatElementsAttr>(operands[0]);
    bool castStatus = true;
    auto elementResult =
        calculate(op.getSplatValue<ElementValueT>(), castStatus);
    if (!castStatus)
      return {};
    auto shapedResType = cast<ShapedType>(resType);
    if (!shapedResType.hasStaticShape())
      return {};
    return DenseElementsAttr::get(shapedResType, elementResult);
  }
  if (auto op = dyn_cast<ElementsAttr>(operands[0])) {
    // Operand is ElementsAttr-derived; perform an element-wise fold by
    // expanding the value.
    bool castStatus = true;
    auto maybeOpIt = op.try_value_begin<ElementValueT>();
    if (!maybeOpIt)
      return {};
    auto opIt = *maybeOpIt;
    SmallVector<TargetElementValueT> elementResults;
    elementResults.reserve(op.getNumElements());
    for (size_t i = 0, e = op.getNumElements(); i < e; ++i, ++opIt) {
      auto elt = calculate(*opIt, castStatus);
      if (!castStatus)
        return {};
      elementResults.push_back(elt);
    }

    return DenseElementsAttr::get(cast<ShapedType>(resType), elementResults);
  }
  return {};
}
} // namespace mlir

#endif // MLIR_DIALECT_COMMONFOLDERS_H
