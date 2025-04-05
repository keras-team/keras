//===- Simplifications.h - Mesh Simplifications -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_TRANSFORMS_SIMPLIFICATIONS_H
#define MLIR_DIALECT_MESH_TRANSFORMS_SIMPLIFICATIONS_H

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/EndomorphismSimplification.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

namespace mlir {

class SymbolTableCollection;

namespace mesh {

// If we have an algebraic op like "+" and a summing all-reduce,
// `all_reduce_sum(x) + all_reduce_sum(y)` will be transformed to
// `all_reduce_sum(x + y)`.
//
// Another example with `min`.
// `min(all_reduce_min(x), all_reduce_min(y))` will be transformed to
// `all_reduce_min(min(x, y))`.
//
// Works only with algebraic ops that have all their operands relevant
// to the all-reduce endomorphism.
// Will not work with some op `f(x, y, z)` where only `x` and `y` form
// the algebraic structure.
template <typename AlgebraicOp>
void populateAllReduceEndomorphismSimplificationPatterns(
    RewritePatternSet &patterns, ReductionKind reduction) {
  auto getEndomorphismOpOperand = [](Operation *op) {
    auto allReduceOp = llvm::cast<AllReduceOp>(op);
    return &allReduceOp.getInputMutable();
  };
  auto getEndomorphismOpResult = [](Operation *op) {
    auto allReduceOp = llvm::cast<AllReduceOp>(op);
    return allReduceOp->getResult(0);
  };
  auto getAlgebraicOpOperands = [](Operation *op,
                                   SmallVector<OpOperand *> &operands) {
    auto algebraicOp = llvm::cast<AlgebraicOp>(op);
    std::transform(algebraicOp->getOpOperands().begin(),
                   algebraicOp->getOpOperands().end(),
                   std::back_inserter(operands),
                   [](OpOperand &operand) { return &operand; });
  };
  auto getAlgebraicOpResult = [](Operation *op) {
    auto algebraicOp = llvm::cast<AlgebraicOp>(op);
    return algebraicOp->getResult(0);
  };
  auto isEndomorphismOp = [reduction](Operation *op,
                                      std::optional<Operation *> referenceOp) {
    auto allReduceOp = llvm::dyn_cast<AllReduceOp>(op);
    if (!allReduceOp ||
        allReduceOp.getInput().getType().getElementType() !=
            allReduceOp.getResult().getType().getElementType() ||
        allReduceOp.getReduction() != reduction) {
      return false;
    }

    // Dont't use simplify if the all-reduce is used other than by the
    // algebraic op.
    // TODO: maybe handle this by an additional pass that later reverses the
    // simplification if there are other uses left other optimizations have
    // been done.
    if (!allReduceOp->hasOneUse()) {
      return false;
    }

    if (!referenceOp) {
      return true;
    }

    auto refAllReduceOp = llvm::dyn_cast<AllReduceOp>(referenceOp.value());
    return refAllReduceOp->getAttrs() == allReduceOp->getAttrs() &&
           allReduceOp.getInput().getType().getElementType() ==
               refAllReduceOp.getInput().getType().getElementType();
  };
  auto isAlgebraicOp = [](Operation *op) {
    return static_cast<bool>(llvm::dyn_cast<AlgebraicOp>(op));
  };

  using ConcreteEndomorphismSimplification = EndomorphismSimplification<
      std::decay_t<decltype(getEndomorphismOpOperand)>,
      std::decay_t<decltype(getEndomorphismOpResult)>,
      std::decay_t<decltype(getAlgebraicOpOperands)>,
      std::decay_t<decltype(getAlgebraicOpResult)>,
      std::decay_t<decltype(isEndomorphismOp)>,
      std::decay_t<decltype(isAlgebraicOp)>>;
  patterns.add(std::make_unique<ConcreteEndomorphismSimplification>(
      std::move(getEndomorphismOpOperand), std::move(getEndomorphismOpResult),
      std::move(getAlgebraicOpOperands), std::move(getAlgebraicOpResult),
      std::move(isEndomorphismOp), std::move(isAlgebraicOp),
      AlgebraicOp::getOperationName(), 1, patterns.getContext()));
}

// It is invalid to change ops that declare symbols during the application of
// these patterns, because symbolTableCollection is used to cache them.
void populateSimplificationPatterns(
    RewritePatternSet &patterns, SymbolTableCollection &symbolTableCollection);
void populateFoldingPatterns(RewritePatternSet &patterns,
                             SymbolTableCollection &symbolTableCollection);

} // namespace mesh
} // namespace mlir

#endif // MLIR_DIALECT_MESH_TRANSFORMS_SIMPLIFICATIONS_H
