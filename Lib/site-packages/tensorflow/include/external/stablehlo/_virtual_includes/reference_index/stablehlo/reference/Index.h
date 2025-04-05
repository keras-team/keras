/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_REFERENCE_INDEX_H
#define STABLEHLO_REFERENCE_INDEX_H

#include <optional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace stablehlo {

class IndexSpaceIterator;

/// Represents per axis metadata (e.g. tensor shape, slice sizes etc.) of type
/// `int64_t`.
class Sizes : public SmallVector<int64_t> {
 public:
  Sizes() = default;
  Sizes(const Sizes &other) = default;
  Sizes &operator=(const Sizes &other) = default;

  Sizes(std::initializer_list<int64_t> list) : SmallVector(list) {}
  Sizes(iterator begin, iterator end) : SmallVector(begin, end) {}
  explicit Sizes(size_t size, int64_t element = 0)
      : SmallVector(size, element) {}
  explicit Sizes(ArrayRef<int64_t> array) : SmallVector(array) {}
  explicit Sizes(DenseIntElementsAttr attr)
      : SmallVector(attr.getValues<int64_t>()) {}

  // Returns `s` with the effect of applying `permutation`
  // to `this` object, that is, `s[i] = (*this)[permutation[i]]`.
  Sizes permute(ArrayRef<int64_t> permutation) const;

  /// Checks if an element `e` at kth axis of `this` object follows
  /// `0 <= e <= bounds[k]`.
  bool inBounds(const Sizes &bounds) const;

  /// Iterate over the index space of a Sizes object.
  IndexSpaceIterator index_begin() const;
  IndexSpaceIterator index_end() const;
};

raw_ostream &operator<<(raw_ostream &os, const Sizes &x);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] + y[k]` for all axis k.
Sizes operator+(const Sizes &x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] + y` for all axis k.
Sizes operator+(const Sizes &x, int64_t y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x + y[k]` for all axis k.
Sizes operator+(int64_t x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] - y[k]` for all axis k.
Sizes operator-(const Sizes &x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] - y` for all axis k.
Sizes operator-(const Sizes &x, int64_t y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x - y[k]` for all axis k.
Sizes operator-(int64_t x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] * y[k]` for all axis k.
Sizes operator*(const Sizes &x, const Sizes &y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x[k] * y` for all axis k.
Sizes operator*(const Sizes &x, int64_t y);

/// Overloaded add operator to return `Sizes` object `z` such that
/// `z[k] = x * y[k]` for all axis k.
Sizes operator*(int64_t x, const Sizes &y);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min[k]), max[k])` for all axis k.
Sizes clamp(const Sizes &min, const Sizes &x, const Sizes &max);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min), max)` for all axis k.
Sizes clamp(int64_t min, const Sizes &x, int64_t max);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min), max[k])` for all axis k.
Sizes clamp(int64_t min, const Sizes &x, const Sizes &max);

/// Clamp operator to return `Sizes` object `z` such that
/// `z[k] = std::min(std::max(x[k], min[k]), max)` for all axis k.
Sizes clamp(const Sizes &min, const Sizes &x, int64_t max);

/// Represents index of a tensor.
using Index = Sizes;

/// Iterates over the index space of a tensor with a given shape, producing
/// indices in lexicographical order. As an example, for a tensor with shape
/// [2,3], the iterator enumerates the indices (0,0), (0,1), (0,2), (1,0),
/// (1,1), (1,2) and <END> (special past-the-end element which cannot be
/// dereferenced).
class IndexSpaceIterator {
 public:
  /// \name Constructor
  IndexSpaceIterator(Sizes shape) : shape_(shape) { index_ = std::nullopt; }

  IndexSpaceIterator(Sizes shape, std::optional<Index> index)
      : shape_(shape), index_(std::nullopt) {
    if (index && index->inBounds(shape)) index_ = index;
  }

  /// Get the current index.
  /// At any point in time, the iterator can either reference an actual index
  /// or the past-the-end element in the index space.
  /// Dereferencing a past-the-end iterator will result in a fatal error.
  const Index &operator*() const;
  const Index *operator->() const;

  /// Compare the iterator to another iterator.
  /// Two iterators are equal if they have the same underlying shape and
  /// reference the same element in the index space.
  bool operator==(const IndexSpaceIterator &it) {
    return shape_ == it.shape_ && index_ == it.index_;
  }
  bool operator!=(const IndexSpaceIterator &it) { return !(*this == it); }

  /// Increment to the next index while iterating over the index space
  /// of a tensor in lexicographical order.
  /// Incrementing past the last index will result in a past-the-end iterator
  /// which cannot be dereferenced. Incrementing even further will result in
  /// a fatal error.
  IndexSpaceIterator &operator++();
  IndexSpaceIterator operator++(int);

 private:
  /// Shape of the tensor whose index space to be iterated on.
  Sizes shape_;

  /// Current multi-dimensional index.
  /// If the optional is empty, then we're at the end
  std::optional<Index> index_;
};

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_INDEX_H
