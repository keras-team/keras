//===- DIExpressionRewriter.h - Rewriter for DIExpression operators -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A driver for running rewrite patterns on DIExpression operators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_DIEXPRESSIONREWRITER_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_DIEXPRESSIONREWRITER_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include <deque>

namespace mlir {
namespace LLVM {

/// Rewriter for DIExpressionAttr.
///
/// Users of this rewriter register their own rewrite patterns. Each pattern
/// matches on a contiguous range of LLVM DIExpressionElemAttrs, and can be
/// used to rewrite it into a new range of DIExpressionElemAttrs of any length.
class DIExpressionRewriter {
public:
  using OperatorT = LLVM::DIExpressionElemAttr;

  class ExprRewritePattern {
  public:
    using OperatorT = DIExpressionRewriter::OperatorT;
    using OpIterT = std::deque<OperatorT>::const_iterator;
    using OpIterRange = llvm::iterator_range<OpIterT>;

    virtual ~ExprRewritePattern() = default;
    /// Checks whether a particular prefix of operators matches this pattern.
    /// The provided argument is guaranteed non-empty.
    /// Return the iterator after the last matched element.
    virtual OpIterT match(OpIterRange) const = 0;
    /// Replace the operators with a new list of operators.
    /// The provided argument is guaranteed to be the same length as returned
    /// by the `match` function.
    virtual SmallVector<OperatorT> replace(OpIterRange) const = 0;
  };

  /// Register a rewrite pattern with the rewriter.
  /// Rewrite patterns are attempted in the order of registration.
  void addPattern(std::unique_ptr<ExprRewritePattern> pattern);

  /// Simplify a DIExpression according to all the patterns registered.
  /// An optional `maxNumRewrites` can be passed to limit the number of rewrites
  /// that gets applied.
  LLVM::DIExpressionAttr
  simplify(LLVM::DIExpressionAttr expr,
           std::optional<uint64_t> maxNumRewrites = {}) const;

private:
  /// The registered patterns.
  SmallVector<std::unique_ptr<ExprRewritePattern>> patterns;
};

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_DIEXPRESSIONREWRITER_H
