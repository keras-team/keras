//===- ByteCode.h - Pattern byte-code interpreter ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a byte-code and interpreter for pattern rewrites in MLIR.
// The byte-code is constructed from the PDL Interpreter dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_BYTECODE_H_
#define MLIR_REWRITE_BYTECODE_H_

#include "mlir/IR/PatternMatch.h"

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH

namespace mlir {
namespace pdl_interp {
class RecordMatchOp;
} // namespace pdl_interp

namespace detail {
class PDLByteCode;

/// Use generic bytecode types. ByteCodeField refers to the actual bytecode
/// entries. ByteCodeAddr refers to size of indices into the bytecode.
using ByteCodeField = uint16_t;
using ByteCodeAddr = uint32_t;
using OwningOpRange = llvm::OwningArrayRef<Operation *>;

//===----------------------------------------------------------------------===//
// PDLByteCodePattern
//===----------------------------------------------------------------------===//

/// All of the data pertaining to a specific pattern within the bytecode.
class PDLByteCodePattern : public Pattern {
public:
  static PDLByteCodePattern create(pdl_interp::RecordMatchOp matchOp,
                                   PDLPatternConfigSet *configSet,
                                   ByteCodeAddr rewriterAddr);

  /// Return the bytecode address of the rewriter for this pattern.
  ByteCodeAddr getRewriterAddr() const { return rewriterAddr; }

  /// Return the configuration set for this pattern, or null if there is none.
  PDLPatternConfigSet *getConfigSet() const { return configSet; }

private:
  template <typename... Args>
  PDLByteCodePattern(ByteCodeAddr rewriterAddr, PDLPatternConfigSet *configSet,
                     Args &&...patternArgs)
      : Pattern(std::forward<Args>(patternArgs)...), rewriterAddr(rewriterAddr),
        configSet(configSet) {}

  /// The address of the rewriter for this pattern.
  ByteCodeAddr rewriterAddr;

  /// The optional config set for this pattern.
  PDLPatternConfigSet *configSet;
};

//===----------------------------------------------------------------------===//
// PDLByteCodeMutableState
//===----------------------------------------------------------------------===//

/// This class contains the mutable state of a bytecode instance. This allows
/// for a bytecode instance to be cached and reused across various different
/// threads/drivers.
class PDLByteCodeMutableState {
public:
  /// Set the new benefit for a bytecode pattern. The `patternIndex` corresponds
  /// to the position of the pattern within the range returned by
  /// `PDLByteCode::getPatterns`.
  void updatePatternBenefit(unsigned patternIndex, PatternBenefit benefit);

  /// Cleanup any allocated state after a match/rewrite has been completed. This
  /// method should be called irregardless of whether the match+rewrite was a
  /// success or not.
  void cleanupAfterMatchAndRewrite();

private:
  /// Allow access to data fields.
  friend class PDLByteCode;

  /// The mutable block of memory used during the matching and rewriting phases
  /// of the bytecode.
  std::vector<const void *> memory;

  /// A mutable block of memory used during the matching and rewriting phase of
  /// the bytecode to store ranges of operations. These are always stored by
  /// owning references, because at no point in the execution of the byte code
  /// we get an indexed range (view) of operations.
  std::vector<OwningOpRange> opRangeMemory;

  /// A mutable block of memory used during the matching and rewriting phase of
  /// the bytecode to store ranges of types.
  std::vector<TypeRange> typeRangeMemory;
  /// A set of type ranges that have been allocated by the byte code interpreter
  /// to provide a guaranteed lifetime.
  std::vector<llvm::OwningArrayRef<Type>> allocatedTypeRangeMemory;

  /// A mutable block of memory used during the matching and rewriting phase of
  /// the bytecode to store ranges of values.
  std::vector<ValueRange> valueRangeMemory;
  /// A set of value ranges that have been allocated by the byte code
  /// interpreter to provide a guaranteed lifetime.
  std::vector<llvm::OwningArrayRef<Value>> allocatedValueRangeMemory;

  /// The current index of ranges being iterated over for each level of nesting.
  /// These are always maintained at 0 for the loops that are not active, so we
  /// do not need to have a separate initialization phase for each loop.
  std::vector<unsigned> loopIndex;

  /// The up-to-date benefits of the patterns held by the bytecode. The order
  /// of this array corresponds 1-1 with the array of patterns in `PDLByteCode`.
  std::vector<PatternBenefit> currentPatternBenefits;
};

//===----------------------------------------------------------------------===//
// PDLByteCode
//===----------------------------------------------------------------------===//

/// The bytecode class is also the interpreter. Contains the bytecode itself,
/// the static info, addresses of the rewriter functions, the interpreter
/// memory buffer, and the execution context.
class PDLByteCode {
public:
  /// Each successful match returns a MatchResult, which contains information
  /// necessary to execute the rewriter and indicates the originating pattern.
  struct MatchResult {
    MatchResult(Location loc, const PDLByteCodePattern &pattern,
                PatternBenefit benefit)
        : location(loc), pattern(&pattern), benefit(benefit) {}
    MatchResult(const MatchResult &) = delete;
    MatchResult &operator=(const MatchResult &) = delete;
    MatchResult(MatchResult &&other) = default;
    MatchResult &operator=(MatchResult &&) = default;

    /// The location of operations to be replaced.
    Location location;
    /// Memory values defined in the matcher that are passed to the rewriter.
    SmallVector<const void *> values;
    /// Memory used for the range input values.
    SmallVector<TypeRange, 0> typeRangeValues;
    SmallVector<ValueRange, 0> valueRangeValues;

    /// The originating pattern that was matched. This is always non-null, but
    /// represented with a pointer to allow for assignment.
    const PDLByteCodePattern *pattern;
    /// The current benefit of the pattern that was matched.
    PatternBenefit benefit;
  };

  /// Create a ByteCode instance from the given module containing operations in
  /// the PDL interpreter dialect.
  PDLByteCode(ModuleOp module,
              SmallVector<std::unique_ptr<PDLPatternConfigSet>> configs,
              const DenseMap<Operation *, PDLPatternConfigSet *> &configMap,
              llvm::StringMap<PDLConstraintFunction> constraintFns,
              llvm::StringMap<PDLRewriteFunction> rewriteFns);

  /// Return the patterns held by the bytecode.
  ArrayRef<PDLByteCodePattern> getPatterns() const { return patterns; }

  /// Initialize the given state such that it can be used to execute the current
  /// bytecode.
  void initializeMutableState(PDLByteCodeMutableState &state) const;

  /// Run the pattern matcher on the given root operation, collecting the
  /// matched patterns in `matches`.
  void match(Operation *op, PatternRewriter &rewriter,
             SmallVectorImpl<MatchResult> &matches,
             PDLByteCodeMutableState &state) const;

  /// Run the rewriter of the given pattern that was previously matched in
  /// `match`. Returns if a failure was encountered during the rewrite.
  LogicalResult rewrite(PatternRewriter &rewriter, const MatchResult &match,
                        PDLByteCodeMutableState &state) const;

private:
  /// Execute the given byte code starting at the provided instruction `inst`.
  /// `matches` is an optional field provided when this function is executed in
  /// a matching context.
  void executeByteCode(const ByteCodeField *inst, PatternRewriter &rewriter,
                       PDLByteCodeMutableState &state,
                       SmallVectorImpl<MatchResult> *matches) const;

  /// The set of pattern configs referenced within the bytecode.
  SmallVector<std::unique_ptr<PDLPatternConfigSet>> configs;

  /// A vector containing pointers to uniqued data. The storage is intentionally
  /// opaque such that we can store a wide range of data types. The types of
  /// data stored here include:
  ///  * Attribute, OperationName, Type
  std::vector<const void *> uniquedData;

  /// A vector containing the generated bytecode for the matcher.
  SmallVector<ByteCodeField, 64> matcherByteCode;

  /// A vector containing the generated bytecode for all of the rewriters.
  SmallVector<ByteCodeField, 64> rewriterByteCode;

  /// The set of patterns contained within the bytecode.
  SmallVector<PDLByteCodePattern, 32> patterns;

  /// A set of user defined functions invoked via PDL.
  std::vector<PDLConstraintFunction> constraintFunctions;
  std::vector<PDLRewriteFunction> rewriteFunctions;

  /// The maximum memory index used by a value.
  ByteCodeField maxValueMemoryIndex = 0;

  /// The maximum number of different types of ranges.
  ByteCodeField maxOpRangeCount = 0;
  ByteCodeField maxTypeRangeCount = 0;
  ByteCodeField maxValueRangeCount = 0;

  /// The maximum number of nested loops.
  ByteCodeField maxLoopLevel = 0;
};

} // namespace detail
} // namespace mlir

#else

namespace mlir::detail {

class PDLByteCodeMutableState {
public:
  void cleanupAfterMatchAndRewrite() {}
  void updatePatternBenefit(unsigned patternIndex, PatternBenefit benefit) {}
};

class PDLByteCodePattern : public Pattern {};

class PDLByteCode {
public:
  struct MatchResult {
    const PDLByteCodePattern *pattern = nullptr;
    PatternBenefit benefit;
  };

  void initializeMutableState(PDLByteCodeMutableState &state) const {}
  void match(Operation *op, PatternRewriter &rewriter,
             SmallVectorImpl<MatchResult> &matches,
             PDLByteCodeMutableState &state) const {}
  LogicalResult rewrite(PatternRewriter &rewriter, const MatchResult &match,
                        PDLByteCodeMutableState &state) const {
    return failure();
  }
  ArrayRef<PDLByteCodePattern> getPatterns() const { return {}; }
};

} // namespace mlir::detail

#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH

#endif // MLIR_REWRITE_BYTECODE_H_
