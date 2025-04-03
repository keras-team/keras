//===- EndomorphismSimplification.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SIMPLIFY_ENDOMORPHISM_H_
#define MLIR_TRANSFORMS_SIMPLIFY_ENDOMORPHISM_H_

#include "mlir/Transforms/HomomorphismSimplification.h"

namespace mlir {

namespace detail {
struct CreateAlgebraicOpForEndomorphismSimplification {
  Operation *operator()(Operation *op, IRMapping &operandsRemapping,
                        PatternRewriter &rewriter) const {
    return rewriter.clone(*op, operandsRemapping);
  }
};
} // namespace detail

// If `f` is an endomorphism with respect to the algebraic structure induced by
// function `g`, transforms `g(f(x1), f(x2) ..., f(xn))` into
// `f(g(x1, x2, ..., xn))`.
// `g` is the algebraic operation and `f` is the endomorphism.
//
// Functors:
// ---------
// `GetEndomorphismOpOperandFn`: `(Operation*) -> OpOperand*`
// Returns the operand relevant to the endomorphism.
// There may be other operands that are not relevant.
//
// `GetEndomorphismOpResultFn`: `(Operation*) -> OpResult`
// Returns the result relevant to the endomorphism.
//
// `GetAlgebraicOpOperandsFn`: `(Operation*, SmallVector<OpOperand*>&) -> void`
// Populates into the vector the operands relevant to the endomorphism.
//
// `GetAlgebraicOpResultFn`: `(Operation*) -> OpResult`
//  Return the result relevant to the endomorphism.
//
// `IsEndomorphismOpFn`: `(Operation*, std::optional<Operation*>) -> bool`
// Check if the operation is an endomorphism of the required type.
// Additionally if the optional is present checks if the operations are
// compatible endomorphisms.
//
// `IsAlgebraicOpFn`: `(Operation*) -> bool`
// Check if the operation is an operation of the algebraic structure.
template <typename GetEndomorphismOpOperandFn,
          typename GetEndomorphismOpResultFn, typename GetAlgebraicOpOperandsFn,
          typename GetAlgebraicOpResultFn, typename IsEndomorphismOpFn,
          typename IsAlgebraicOpFn>
struct EndomorphismSimplification
    : HomomorphismSimplification<
          GetEndomorphismOpOperandFn, GetEndomorphismOpResultFn,
          GetAlgebraicOpOperandsFn, GetAlgebraicOpResultFn,
          GetAlgebraicOpResultFn, IsEndomorphismOpFn, IsAlgebraicOpFn,
          detail::CreateAlgebraicOpForEndomorphismSimplification> {
  template <typename GetEndomorphismOpOperandFnArg,
            typename GetEndomorphismOpResultFnArg,
            typename GetAlgebraicOpOperandsFnArg,
            typename GetAlgebraicOpResultFnArg, typename IsEndomorphismOpFnArg,
            typename IsAlgebraicOpFnArg, typename... RewritePatternArgs>
  EndomorphismSimplification(
      GetEndomorphismOpOperandFnArg &&getEndomorphismOpOperand,
      GetEndomorphismOpResultFnArg &&getEndomorphismOpResult,
      GetAlgebraicOpOperandsFnArg &&getAlgebraicOpOperands,
      GetAlgebraicOpResultFnArg &&getAlgebraicOpResult,
      IsEndomorphismOpFnArg &&isEndomorphismOp,
      IsAlgebraicOpFnArg &&isAlgebraicOp, RewritePatternArgs &&...args)
      : HomomorphismSimplification<
            GetEndomorphismOpOperandFn, GetEndomorphismOpResultFn,
            GetAlgebraicOpOperandsFn, GetAlgebraicOpResultFn,
            GetAlgebraicOpResultFn, IsEndomorphismOpFn, IsAlgebraicOpFn,
            detail::CreateAlgebraicOpForEndomorphismSimplification>(
            std::forward<GetEndomorphismOpOperandFnArg>(
                getEndomorphismOpOperand),
            std::forward<GetEndomorphismOpResultFnArg>(getEndomorphismOpResult),
            std::forward<GetAlgebraicOpOperandsFnArg>(getAlgebraicOpOperands),
            std::forward<GetAlgebraicOpResultFnArg>(getAlgebraicOpResult),
            std::forward<GetAlgebraicOpResultFnArg>(getAlgebraicOpResult),
            std::forward<IsEndomorphismOpFnArg>(isEndomorphismOp),
            std::forward<IsAlgebraicOpFnArg>(isAlgebraicOp),
            detail::CreateAlgebraicOpForEndomorphismSimplification(),
            std::forward<RewritePatternArgs>(args)...) {}
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_SIMPLIFY_ENDOMORPHISM_H_
