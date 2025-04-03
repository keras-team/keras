//===- DIExpressionLegalization.h - DIExpression Legalization Patterns ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for known legalization patterns for DIExpressions that should
// be performed before translation into llvm.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_DIEXPRESSIONLEGALIZATION_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_DIEXPRESSIONLEGALIZATION_H

#include "mlir/Dialect/LLVMIR/Transforms/DIExpressionRewriter.h"

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

/// Adjacent DW_OP_LLVM_fragment should be merged into one.
///
/// E.g.
///   #llvm.di_expression<[
///     DW_OP_LLVM_fragment(32, 32), DW_OP_LLVM_fragment(32, 64)
///   ]>
/// =>
///   #llvm.di_expression<[DW_OP_LLVM_fragment(64, 32)]>
class MergeFragments : public DIExpressionRewriter::ExprRewritePattern {
public:
  OpIterT match(OpIterRange operators) const override;
  SmallVector<OperatorT> replace(OpIterRange operators) const override;
};

//===----------------------------------------------------------------------===//
// Runner
//===----------------------------------------------------------------------===//

/// Register all known legalization patterns declared here and apply them to
/// all ops in `op`.
void legalizeDIExpressionsRecursively(Operation *op);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_DIEXPRESSIONLEGALIZATION_H
