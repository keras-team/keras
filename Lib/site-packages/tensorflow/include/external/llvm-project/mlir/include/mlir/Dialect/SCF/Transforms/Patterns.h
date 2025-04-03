//===- Patterns.h - SCF dialect rewrite patterns ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_TRANSFORMS_PATTERNS_H
#define MLIR_DIALECT_SCF_TRANSFORMS_PATTERNS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

class ConversionTarget;
class TypeConverter;

namespace scf {

// TODO: such patterns should be auto-generated.
class ForLoopPipeliningPattern : public OpRewritePattern<ForOp> {
public:
  ForLoopPipeliningPattern(const PipeliningOption &options,
                           MLIRContext *context)
      : OpRewritePattern<ForOp>(context), options(options) {}
  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(forOp, rewriter);
  }

  FailureOr<ForOp> returningMatchAndRewrite(ForOp forOp,
                                            PatternRewriter &rewriter) const {
    return pipelineForLoop(rewriter, forOp, options);
  }

protected:
  PipeliningOption options;
};

/// Populates patterns for SCF structural type conversions and sets up the
/// provided ConversionTarget with the appropriate legality configuration for
/// the ops to get converted properly.
///
/// A "structural" type conversion is one where the underlying ops are
/// completely agnostic to the actual types involved and simply need to update
/// their types. An example of this is scf.if -- the scf.if op and the
/// corresponding scf.yield ops need to update their types accordingly to the
/// TypeConverter, but otherwise don't care what type conversions are happening.
void populateSCFStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target);

/// Similar to `populateSCFStructuralTypeConversionsAndLegality` but does not
/// populate the conversion target.
void populateSCFStructuralTypeConversions(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);

/// Updates the ConversionTarget with dynamic legality of SCF operations based
/// on the provided type converter.
void populateSCFStructuralTypeConversionTarget(
    const TypeConverter &typeConverter, ConversionTarget &target);

/// Populates the provided pattern set with patterns that do 1:N type
/// conversions on (some) SCF ops. This is intended to be used with
/// applyPartialOneToNConversion.
void populateSCFStructuralOneToNTypeConversions(TypeConverter &typeConverter,
                                                RewritePatternSet &patterns);

/// Populate patterns for SCF software pipelining transformation. See the
/// ForLoopPipeliningPattern for the transformation details.
void populateSCFLoopPipeliningPatterns(RewritePatternSet &patterns,
                                       const PipeliningOption &options);

/// Populate patterns for canonicalizing operations inside SCF loop bodies.
/// At the moment, only affine.min/max computations with iteration variables,
/// loop bounds and loop steps are canonicalized.
void populateSCFForLoopCanonicalizationPatterns(RewritePatternSet &patterns);

/// Populate patterns to uplift `scf.while` ops to `scf.for`.
/// Uplifitng expects a specific ops pattern:
///  * `before` block consisting of single arith.cmp op
///  * `after` block containing arith.addi
void populateUpliftWhileToForPatterns(RewritePatternSet &patterns);

/// Populate patterns to rotate `scf.while` ops, constructing `do-while` loops
/// from `while` loops.
void populateSCFRotateWhileLoopPatterns(RewritePatternSet &patterns);
} // namespace scf
} // namespace mlir

#endif // MLIR_DIALECT_SCF_TRANSFORMS_PATTERNS_H
