//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_ARITH_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class DataFlowSolver;
class ConversionTarget;
class TypeConverter;

namespace arith {

#define GEN_PASS_DECL
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"

class WideIntEmulationConverter;
class NarrowTypeEmulationConverter;

/// Adds patterns to emulate wide Arith and Function ops over integer
/// types into supported ones. This is done by splitting original power-of-two
/// i2N integer types into two iN halves.
void populateArithWideIntEmulationPatterns(
    WideIntEmulationConverter &typeConverter, RewritePatternSet &patterns);

/// Adds patterns to emulate narrow Arith and Function ops into wide
/// supported types. Users need to add conversions about the computation
/// domain of narrow types.
void populateArithNarrowTypeEmulationPatterns(
    NarrowTypeEmulationConverter &typeConverter, RewritePatternSet &patterns);

/// Populate the type conversions needed to emulate the unsupported
/// `sourceTypes` with `destType`
void populateEmulateUnsupportedFloatsConversions(TypeConverter &converter,
                                                 ArrayRef<Type> sourceTypes,
                                                 Type targetType);

/// Add rewrite patterns for converting operations that use illegal float types
/// to ones that use legal ones.
void populateEmulateUnsupportedFloatsPatterns(RewritePatternSet &patterns,
                                              TypeConverter &converter);

/// Set up a dialect conversion to reject arithmetic operations on unsupported
/// float types.
void populateEmulateUnsupportedFloatsLegality(ConversionTarget &target,
                                              TypeConverter &converter);
/// Add patterns to expand Arith ceil/floor division ops.
void populateCeilFloorDivExpandOpsPatterns(RewritePatternSet &patterns);

/// Add patterns to expand Arith bf16 patterns to lower level bitcasts/shifts.
void populateExpandBFloat16Patterns(RewritePatternSet &patterns);

/// Add patterns to expand Arith ops.
void populateArithExpandOpsPatterns(RewritePatternSet &patterns);

/// Create a pass to replace signed ops with unsigned ones where they are proven
/// equivalent.
std::unique_ptr<Pass> createArithUnsignedWhenEquivalentPass();

/// Add patterns for int range based optimizations.
void populateIntRangeOptimizationsPatterns(RewritePatternSet &patterns,
                                           DataFlowSolver &solver);

/// Create a pass which do optimizations based on integer range analysis.
std::unique_ptr<Pass> createIntRangeOptimizationsPass();

/// Add patterns for integer bitwidth narrowing.
void populateArithIntNarrowingPatterns(RewritePatternSet &patterns,
                                       const ArithIntNarrowingOptions &options);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Arith/Transforms/Passes.h.inc"

} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_TRANSFORMS_PASSES_H_
