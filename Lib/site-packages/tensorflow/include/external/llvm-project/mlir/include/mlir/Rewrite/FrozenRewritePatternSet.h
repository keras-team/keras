//===- FrozenRewritePatternSet.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_FROZENREWRITEPATTERNSET_H
#define MLIR_REWRITE_FROZENREWRITEPATTERNSET_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace detail {
class PDLByteCode;
} // namespace detail

/// This class represents a frozen set of patterns that can be processed by a
/// pattern applicator. This class is designed to enable caching pattern lists
/// such that they need not be continuously recomputed. Note that all copies of
/// this class share the same compiled pattern list, allowing for a reduction in
/// the number of duplicated patterns that need to be created.
class FrozenRewritePatternSet {
  using NativePatternListT = std::vector<std::unique_ptr<RewritePattern>>;

public:
  /// A map of operation specific native patterns.
  using OpSpecificNativePatternListT =
      DenseMap<OperationName, std::vector<RewritePattern *>>;

  FrozenRewritePatternSet();
  FrozenRewritePatternSet(FrozenRewritePatternSet &&patterns) = default;
  FrozenRewritePatternSet(const FrozenRewritePatternSet &patterns) = default;
  FrozenRewritePatternSet &
  operator=(const FrozenRewritePatternSet &patterns) = default;
  FrozenRewritePatternSet &
  operator=(FrozenRewritePatternSet &&patterns) = default;
  ~FrozenRewritePatternSet();

  /// Freeze the patterns held in `patterns`, and take ownership.
  /// `disabledPatternLabels` is a set of labels used to filter out input
  /// patterns with a debug label or debug name in this set.
  /// `enabledPatternLabels` is a set of labels used to filter out input
  /// patterns that do not have one of the labels in this set. Debug labels must
  /// be set explicitly on patterns or when adding them with
  /// `RewritePatternSet::addWithLabel`. Debug names may be empty, but patterns
  /// created with `RewritePattern::create` have their default debug name set to
  /// their type name.
  FrozenRewritePatternSet(
      RewritePatternSet &&patterns,
      ArrayRef<std::string> disabledPatternLabels = std::nullopt,
      ArrayRef<std::string> enabledPatternLabels = std::nullopt);

  /// Return the op specific native patterns held by this list.
  const OpSpecificNativePatternListT &getOpSpecificNativePatterns() const {
    return impl->nativeOpSpecificPatternMap;
  }

  /// Return the "match any" native patterns held by this list.
  iterator_range<llvm::pointee_iterator<NativePatternListT::const_iterator>>
  getMatchAnyOpNativePatterns() const {
    const NativePatternListT &nativeList = impl->nativeAnyOpPatterns;
    return llvm::make_pointee_range(nativeList);
  }

  /// Return the compiled PDL bytecode held by this list. Returns null if
  /// there are no PDL patterns within the list.
  const detail::PDLByteCode *getPDLByteCode() const {
    return impl->pdlByteCode.get();
  }

private:
  /// The internal implementation of the frozen pattern list.
  struct Impl {
    /// The set of native C++ rewrite patterns that are matched to specific
    /// operation kinds.
    OpSpecificNativePatternListT nativeOpSpecificPatternMap;

    /// The full op-specific native rewrite list. This allows for the map above
    /// to contain duplicate patterns, e.g. for interfaces and traits.
    NativePatternListT nativeOpSpecificPatternList;

    /// The set of native C++ rewrite patterns that are matched to "any"
    /// operation.
    NativePatternListT nativeAnyOpPatterns;

    /// The bytecode containing the compiled PDL patterns.
    std::unique_ptr<detail::PDLByteCode> pdlByteCode;
  };

  /// A pointer to the internal pattern list. This uses a shared_ptr to avoid
  /// the need to compile the same pattern list multiple times. For example,
  /// during multi-threaded pass execution, all copies of a pass can share the
  /// same pattern list.
  std::shared_ptr<Impl> impl;
};

} // namespace mlir

#endif // MLIR_REWRITE_FROZENREWRITEPATTERNSET_H
