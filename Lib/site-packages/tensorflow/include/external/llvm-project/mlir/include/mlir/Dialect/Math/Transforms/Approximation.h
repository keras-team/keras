//===- Approximation.h - Math dialect -----------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MATH_TRANSFORMS_APPROXIMATION_H
#define MLIR_DIALECT_MATH_TRANSFORMS_APPROXIMATION_H

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace math {

struct ErfPolynomialApproximation : public OpRewritePattern<math::ErfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ErfOp op,
                                PatternRewriter &rewriter) const final;
};

} // namespace math
} // namespace mlir

#endif // MLIR_DIALECT_MATH_TRANSFORMS_APPROXIMATION_H
