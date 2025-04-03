//===- CyclicReplacerCache.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains helper classes for caching replacer-like functions that
// map values between two domains. They are able to handle replacer logic that
// contains self-recursion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_CYCLICREPLACERCACHE_H
#define MLIR_SUPPORT_CYCLICREPLACERCACHE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <optional>
#include <set>

namespace mlir {

//===----------------------------------------------------------------------===//
// CyclicReplacerCache
//===----------------------------------------------------------------------===//

/// A cache for replacer-like functions that map values between two domains. The
/// difference compared to just using a map to cache in-out pairs is that this
/// class is able to handle replacer logic that is self-recursive (and thus may
/// cause infinite recursion in the naive case).
///
/// This class provides a hook for the user to perform cycle pruning when a
/// cycle is identified, and is able to perform context-sensitive caching so
/// that the replacement result for an input that is part of a pruned cycle can
/// be distinct from the replacement result for the same input when it is not
/// part of a cycle.
///
/// In addition, this class allows deferring cycle pruning until specific inputs
/// are repeated. This is useful for cases where not all elements in a cycle can
/// perform pruning. The user still must guarantee that at least one element in
/// any given cycle can perform pruning. Even if not, an assertion will
/// eventually be tripped instead of infinite recursion (the run-time is
/// linearly bounded by the maximum cycle length of its input).
///
/// WARNING: This class works best with InT & OutT that are trivial scalar
/// types. The input/output elements will be frequently copied and hashed.
template <typename InT, typename OutT>
class CyclicReplacerCache {
public:
  /// User-provided replacement function & cycle-breaking functions.
  /// The cycle-breaking function must not make any more recursive invocations
  /// to this cached replacer.
  using CycleBreakerFn = std::function<std::optional<OutT>(InT)>;

  CyclicReplacerCache() = delete;
  CyclicReplacerCache(CycleBreakerFn cycleBreaker)
      : cycleBreaker(std::move(cycleBreaker)) {}

  /// A possibly unresolved cache entry.
  /// If unresolved, the entry must be resolved before it goes out of scope.
  struct CacheEntry {
  public:
    ~CacheEntry() { assert(result && "unresovled cache entry"); }

    /// Check whether this node was repeated during recursive replacements.
    /// This only makes sense to be called after all recursive replacements are
    /// completed and the current element has resurfaced to the top of the
    /// replacement stack.
    bool wasRepeated() const {
      // If the top frame includes itself as a dependency, then it must have
      // been repeated.
      ReplacementFrame &currFrame = cache.replacementStack.back();
      size_t currFrameIndex = cache.replacementStack.size() - 1;
      return currFrame.dependentFrames.count(currFrameIndex);
    }

    /// Resolve an unresolved cache entry by providing the result to be stored
    /// in the cache.
    void resolve(OutT result) {
      assert(!this->result && "cache entry already resolved");
      cache.finalizeReplacement(element, result);
      this->result = std::move(result);
    }

    /// Get the resolved result if one exists.
    const std::optional<OutT> &get() const { return result; }

  private:
    friend class CyclicReplacerCache;
    CacheEntry() = delete;
    CacheEntry(CyclicReplacerCache<InT, OutT> &cache, InT element,
               std::optional<OutT> result = std::nullopt)
        : cache(cache), element(std::move(element)), result(result) {}

    CyclicReplacerCache<InT, OutT> &cache;
    InT element;
    std::optional<OutT> result;
  };

  /// Lookup the cache for a pre-calculated replacement for `element`.
  /// If one exists, a resolved CacheEntry will be returned. Otherwise, an
  /// unresolved CacheEntry will be returned, and the caller must resolve it
  /// with the calculated replacement so it can be registered in the cache for
  /// future use.
  /// Multiple unresolved CacheEntries may be retrieved. However, any unresolved
  /// CacheEntries that are returned must be resolved in reverse order of
  /// retrieval, i.e. the last retrieved CacheEntry must be resolved first, and
  /// the first retrieved CacheEntry must be resolved last. This should be
  /// natural when used as a stack / inside recursion.
  CacheEntry lookupOrInit(InT element);

private:
  /// Register the replacement in the cache and update the replacementStack.
  void finalizeReplacement(InT element, OutT result);

  CycleBreakerFn cycleBreaker;
  llvm::DenseMap<InT, OutT> standaloneCache;

  struct DependentReplacement {
    OutT replacement;
    /// The highest replacement frame index that this cache entry is dependent
    /// on.
    size_t highestDependentFrame;
  };
  llvm::DenseMap<InT, DependentReplacement> dependentCache;

  struct ReplacementFrame {
    /// The set of elements that is only legal while under this current frame.
    /// They need to be removed from the cache when this frame is popped off the
    /// replacement stack.
    llvm::DenseSet<InT> dependingReplacements;
    /// The set of frame indices that this current frame's replacement is
    /// dependent on, ordered from highest to lowest.
    std::set<size_t, std::greater<size_t>> dependentFrames;
  };
  /// Every element currently in the progress of being replaced pushes a frame
  /// onto this stack.
  llvm::SmallVector<ReplacementFrame> replacementStack;
  /// Maps from each input element to its indices on the replacement stack.
  llvm::DenseMap<InT, llvm::SmallVector<size_t, 2>> cyclicElementFrame;
  /// If set to true, we are currently asking an element to break a cycle. No
  /// more recursive invocations is allowed while this is true (the replacement
  /// stack can no longer grow).
  bool resolvingCycle = false;
};

template <typename InT, typename OutT>
typename CyclicReplacerCache<InT, OutT>::CacheEntry
CyclicReplacerCache<InT, OutT>::lookupOrInit(InT element) {
  assert(!resolvingCycle &&
         "illegal recursive invocation while breaking cycle");

  if (auto it = standaloneCache.find(element); it != standaloneCache.end())
    return CacheEntry(*this, element, it->second);

  if (auto it = dependentCache.find(element); it != dependentCache.end()) {
    // Update the current top frame (the element that invoked this current
    // replacement) to include any dependencies the cache entry had.
    ReplacementFrame &currFrame = replacementStack.back();
    currFrame.dependentFrames.insert(it->second.highestDependentFrame);
    return CacheEntry(*this, element, it->second.replacement);
  }

  auto [it, inserted] = cyclicElementFrame.try_emplace(element);
  if (!inserted) {
    // This is a repeat of a known element. Try to break cycle here.
    resolvingCycle = true;
    std::optional<OutT> result = cycleBreaker(element);
    resolvingCycle = false;
    if (result) {
      // Cycle was broken.
      size_t dependentFrame = it->second.back();
      dependentCache[element] = {*result, dependentFrame};
      ReplacementFrame &currFrame = replacementStack.back();
      // If this is a repeat, there is no replacement frame to pop. Mark the top
      // frame as being dependent on this element.
      currFrame.dependentFrames.insert(dependentFrame);

      return CacheEntry(*this, element, *result);
    }

    // Cycle could not be broken.
    // A legal setup must ensure at least one element of each cycle can break
    // cycles. Under this setup, each element can be seen at most twice before
    // the cycle is broken. If we see an element more than twice, we know this
    // is an illegal setup.
    assert(it->second.size() <= 2 && "illegal 3rd repeat of input");
  }

  // Otherwise, either this is the first time we see this element, or this
  // element could not break this cycle.
  it->second.push_back(replacementStack.size());
  replacementStack.emplace_back();

  return CacheEntry(*this, element);
}

template <typename InT, typename OutT>
void CyclicReplacerCache<InT, OutT>::finalizeReplacement(InT element,
                                                         OutT result) {
  ReplacementFrame &currFrame = replacementStack.back();
  // With the conclusion of this replacement frame, the current element is no
  // longer a dependent element.
  currFrame.dependentFrames.erase(replacementStack.size() - 1);

  auto prevLayerIter = ++replacementStack.rbegin();
  if (prevLayerIter == replacementStack.rend()) {
    // If this is the last frame, there should be zero dependents.
    assert(currFrame.dependentFrames.empty() &&
           "internal error: top-level dependent replacement");
    // Cache standalone result.
    standaloneCache[element] = result;
  } else if (currFrame.dependentFrames.empty()) {
    // Cache standalone result.
    standaloneCache[element] = result;
  } else {
    // Cache dependent result.
    size_t highestDependentFrame = *currFrame.dependentFrames.begin();
    dependentCache[element] = {result, highestDependentFrame};

    // Otherwise, the previous frame inherits the same dependent frames.
    prevLayerIter->dependentFrames.insert(currFrame.dependentFrames.begin(),
                                          currFrame.dependentFrames.end());

    // Mark this current replacement as a depending replacement on the closest
    // dependent frame.
    replacementStack[highestDependentFrame].dependingReplacements.insert(
        element);
  }

  // All depending replacements in the cache must be purged.
  for (InT key : currFrame.dependingReplacements)
    dependentCache.erase(key);

  replacementStack.pop_back();
  auto it = cyclicElementFrame.find(element);
  it->second.pop_back();
  if (it->second.empty())
    cyclicElementFrame.erase(it);
}

//===----------------------------------------------------------------------===//
// CachedCyclicReplacer
//===----------------------------------------------------------------------===//

/// A helper class for cases where the input/output types of the replacer
/// function is identical to the types stored in the cache. This class wraps
/// the user-provided replacer function, and can be used in place of the user
/// function.
template <typename InT, typename OutT>
class CachedCyclicReplacer {
public:
  using ReplacerFn = std::function<OutT(InT)>;
  using CycleBreakerFn =
      typename CyclicReplacerCache<InT, OutT>::CycleBreakerFn;

  CachedCyclicReplacer() = delete;
  CachedCyclicReplacer(ReplacerFn replacer, CycleBreakerFn cycleBreaker)
      : replacer(std::move(replacer)), cache(std::move(cycleBreaker)) {}

  OutT operator()(InT element) {
    auto cacheEntry = cache.lookupOrInit(element);
    if (std::optional<OutT> result = cacheEntry.get())
      return *result;

    OutT result = replacer(element);
    cacheEntry.resolve(result);
    return result;
  }

private:
  ReplacerFn replacer;
  CyclicReplacerCache<InT, OutT> cache;
};

} // namespace mlir

#endif // MLIR_SUPPORT_CYCLICREPLACERCACHE_H
