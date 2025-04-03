//===- HomomorphismSimplification.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SIMPLIFY_HOMOMORPHISM_H_
#define MLIR_TRANSFORMS_SIMPLIFY_HOMOMORPHISM_H_

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir {

// If `h` is an homomorphism with respect to the source algebraic structure
// induced by function `s` and the target algebraic structure induced by
// function `t`, transforms `s(h(x1), h(x2) ..., h(xn))` into
// `h(t(x1, x2, ..., xn))`.
//
// Functors:
// ---------
// `GetHomomorphismOpOperandFn`: `(Operation*) -> OpOperand*`
// Returns the operand relevant to the homomorphism.
// There may be other operands that are not relevant.
//
// `GetHomomorphismOpResultFn`: `(Operation*) -> OpResult`
// Returns the result relevant to the homomorphism.
//
// `GetSourceAlgebraicOpOperandsFn`: `(Operation*, SmallVector<OpOperand*>&) ->
// void` Populates into the vector the operands relevant to the homomorphism.
//
// `GetSourceAlgebraicOpResultFn`: `(Operation*) -> OpResult`
//  Return the result of the source algebraic operation relevant to the
//  homomorphism.
//
// `GetTargetAlgebraicOpResultFn`: `(Operation*) -> OpResult`
//  Return the result of the target algebraic operation relevant to the
//  homomorphism.
//
// `IsHomomorphismOpFn`: `(Operation*, std::optional<Operation*>) -> bool`
// Check if the operation is an homomorphism of the required type.
// Additionally if the optional is present checks if the operations are
// compatible homomorphisms.
//
// `IsSourceAlgebraicOpFn`: `(Operation*) -> bool`
// Check if the operation is an operation of the algebraic structure.
//
// `CreateTargetAlgebraicOpFn`: `(Operation*, IRMapping& operandsRemapping,
// PatternRewriter &rewriter) -> Operation*`
template <typename GetHomomorphismOpOperandFn,
          typename GetHomomorphismOpResultFn,
          typename GetSourceAlgebraicOpOperandsFn,
          typename GetSourceAlgebraicOpResultFn,
          typename GetTargetAlgebraicOpResultFn, typename IsHomomorphismOpFn,
          typename IsSourceAlgebraicOpFn, typename CreateTargetAlgebraicOpFn>
struct HomomorphismSimplification : public RewritePattern {
  template <typename GetHomomorphismOpOperandFnArg,
            typename GetHomomorphismOpResultFnArg,
            typename GetSourceAlgebraicOpOperandsFnArg,
            typename GetSourceAlgebraicOpResultFnArg,
            typename GetTargetAlgebraicOpResultFnArg,
            typename IsHomomorphismOpFnArg, typename IsSourceAlgebraicOpFnArg,
            typename CreateTargetAlgebraicOpFnArg,
            typename... RewritePatternArgs>
  HomomorphismSimplification(
      GetHomomorphismOpOperandFnArg &&getHomomorphismOpOperand,
      GetHomomorphismOpResultFnArg &&getHomomorphismOpResult,
      GetSourceAlgebraicOpOperandsFnArg &&getSourceAlgebraicOpOperands,
      GetSourceAlgebraicOpResultFnArg &&getSourceAlgebraicOpResult,
      GetTargetAlgebraicOpResultFnArg &&getTargetAlgebraicOpResult,
      IsHomomorphismOpFnArg &&isHomomorphismOp,
      IsSourceAlgebraicOpFnArg &&isSourceAlgebraicOp,
      CreateTargetAlgebraicOpFnArg &&createTargetAlgebraicOpFn,
      RewritePatternArgs &&...args)
      : RewritePattern(std::forward<RewritePatternArgs>(args)...),
        getHomomorphismOpOperand(std::forward<GetHomomorphismOpOperandFnArg>(
            getHomomorphismOpOperand)),
        getHomomorphismOpResult(std::forward<GetHomomorphismOpResultFnArg>(
            getHomomorphismOpResult)),
        getSourceAlgebraicOpOperands(
            std::forward<GetSourceAlgebraicOpOperandsFnArg>(
                getSourceAlgebraicOpOperands)),
        getSourceAlgebraicOpResult(
            std::forward<GetSourceAlgebraicOpResultFnArg>(
                getSourceAlgebraicOpResult)),
        getTargetAlgebraicOpResult(
            std::forward<GetTargetAlgebraicOpResultFnArg>(
                getTargetAlgebraicOpResult)),
        isHomomorphismOp(std::forward<IsHomomorphismOpFnArg>(isHomomorphismOp)),
        isSourceAlgebraicOp(
            std::forward<IsSourceAlgebraicOpFnArg>(isSourceAlgebraicOp)),
        createTargetAlgebraicOpFn(std::forward<CreateTargetAlgebraicOpFnArg>(
            createTargetAlgebraicOpFn)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpOperand *> algebraicOpOperands;
    if (failed(matchOp(op, algebraicOpOperands))) {
      return failure();
    }
    return rewriteOp(op, algebraicOpOperands, rewriter);
  }

private:
  LogicalResult
  matchOp(Operation *sourceAlgebraicOp,
          SmallVector<OpOperand *> &sourceAlgebraicOpOperands) const {
    if (!isSourceAlgebraicOp(sourceAlgebraicOp)) {
      return failure();
    }
    sourceAlgebraicOpOperands.clear();
    getSourceAlgebraicOpOperands(sourceAlgebraicOp, sourceAlgebraicOpOperands);
    if (sourceAlgebraicOpOperands.empty()) {
      return failure();
    }

    Operation *firstHomomorphismOp =
        sourceAlgebraicOpOperands.front()->get().getDefiningOp();
    if (!firstHomomorphismOp ||
        !isHomomorphismOp(firstHomomorphismOp, std::nullopt)) {
      return failure();
    }
    OpResult firstHomomorphismOpResult =
        getHomomorphismOpResult(firstHomomorphismOp);
    if (firstHomomorphismOpResult != sourceAlgebraicOpOperands.front()->get()) {
      return failure();
    }

    for (auto operand : sourceAlgebraicOpOperands) {
      Operation *homomorphismOp = operand->get().getDefiningOp();
      if (!homomorphismOp ||
          !isHomomorphismOp(homomorphismOp, firstHomomorphismOp)) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult
  rewriteOp(Operation *sourceAlgebraicOp,
            const SmallVector<OpOperand *> &sourceAlgebraicOpOperands,
            PatternRewriter &rewriter) const {
    IRMapping irMapping;
    for (auto operand : sourceAlgebraicOpOperands) {
      Operation *homomorphismOp = operand->get().getDefiningOp();
      irMapping.map(operand->get(),
                    getHomomorphismOpOperand(homomorphismOp)->get());
    }
    Operation *targetAlgebraicOp =
        createTargetAlgebraicOpFn(sourceAlgebraicOp, irMapping, rewriter);

    irMapping.clear();
    assert(!sourceAlgebraicOpOperands.empty());
    Operation *firstHomomorphismOp =
        sourceAlgebraicOpOperands[0]->get().getDefiningOp();
    irMapping.map(getHomomorphismOpOperand(firstHomomorphismOp)->get(),
                  getTargetAlgebraicOpResult(targetAlgebraicOp));
    Operation *newHomomorphismOp =
        rewriter.clone(*firstHomomorphismOp, irMapping);
    rewriter.replaceAllUsesWith(getSourceAlgebraicOpResult(sourceAlgebraicOp),
                                getHomomorphismOpResult(newHomomorphismOp));
    return success();
  }

  GetHomomorphismOpOperandFn getHomomorphismOpOperand;
  GetHomomorphismOpResultFn getHomomorphismOpResult;
  GetSourceAlgebraicOpOperandsFn getSourceAlgebraicOpOperands;
  GetSourceAlgebraicOpResultFn getSourceAlgebraicOpResult;
  GetTargetAlgebraicOpResultFn getTargetAlgebraicOpResult;
  IsHomomorphismOpFn isHomomorphismOp;
  IsSourceAlgebraicOpFn isSourceAlgebraicOp;
  CreateTargetAlgebraicOpFn createTargetAlgebraicOpFn;
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_SIMPLIFY_HOMOMORPHISM_H_
