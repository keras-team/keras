//===- PatternApplicator.h - PatternApplicator ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an applicator that applies pattern rewrites based upon a
// user defined cost model.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_PATTERNAPPLICATOR_H
#define MLIR_REWRITE_PATTERNAPPLICATOR_H

#include "mlir/Rewrite/FrozenRewritePatternSet.h"

#include "mlir/IR/Action.h"

namespace mlir {
class PatternRewriter;

namespace detail {
class PDLByteCodeMutableState;
} // namespace detail

/// This is the type of Action that is dispatched when a pattern is applied.
/// It captures the pattern to apply on top of the usual context.
class ApplyPatternAction : public tracing::ActionImpl<ApplyPatternAction> {
public:
  using Base = tracing::ActionImpl<ApplyPatternAction>;
  ApplyPatternAction(ArrayRef<IRUnit> irUnits, const Pattern &pattern)
      : Base(irUnits), pattern(pattern) {}
  static constexpr StringLiteral tag = "apply-pattern";
  static constexpr StringLiteral desc =
      "Encapsulate the application of rewrite patterns";

  void print(raw_ostream &os) const override {
    os << "`" << tag << " pattern: " << pattern.getDebugName();
  }

private:
  const Pattern &pattern;
};

/// This class manages the application of a group of rewrite patterns, with a
/// user-provided cost model.
class PatternApplicator {
public:
  /// The cost model dynamically assigns a PatternBenefit to a particular
  /// pattern. Users can query contained patterns and pass analysis results to
  /// applyCostModel. Patterns to be discarded should have a benefit of
  /// `impossibleToMatch`.
  using CostModel = function_ref<PatternBenefit(const Pattern &)>;

  explicit PatternApplicator(const FrozenRewritePatternSet &frozenPatternList);
  ~PatternApplicator();

  /// Attempt to match and rewrite the given op with any pattern, allowing a
  /// predicate to decide if a pattern can be applied or not, and hooks for if
  /// the pattern match was a success or failure.
  ///
  /// canApply:  called before each match and rewrite attempt; return false to
  ///            skip pattern.
  /// onFailure: called when a pattern fails to match to perform cleanup.
  /// onSuccess: called when a pattern match succeeds; return failure() to
  ///            invalidate the match and try another pattern.
  LogicalResult
  matchAndRewrite(Operation *op, PatternRewriter &rewriter,
                  function_ref<bool(const Pattern &)> canApply = {},
                  function_ref<void(const Pattern &)> onFailure = {},
                  function_ref<LogicalResult(const Pattern &)> onSuccess = {});

  /// Apply a cost model to the patterns within this applicator.
  void applyCostModel(CostModel model);

  /// Apply the default cost model that solely uses the pattern's static
  /// benefit.
  void applyDefaultCostModel() {
    applyCostModel([](const Pattern &pattern) { return pattern.getBenefit(); });
  }

  /// Walk all of the patterns within the applicator.
  void walkAllPatterns(function_ref<void(const Pattern &)> walk);

private:
  /// The list that owns the patterns used within this applicator.
  const FrozenRewritePatternSet &frozenPatternList;
  /// The set of patterns to match for each operation, stable sorted by benefit.
  DenseMap<OperationName, SmallVector<const RewritePattern *, 2>> patterns;
  /// The set of patterns that may match against any operation type, stable
  /// sorted by benefit.
  SmallVector<const RewritePattern *, 1> anyOpPatterns;
  /// The mutable state used during execution of the PDL bytecode.
  std::unique_ptr<detail::PDLByteCodeMutableState> mutableByteCodeState;
};

} // namespace mlir

#endif // MLIR_REWRITE_PATTERNAPPLICATOR_H
