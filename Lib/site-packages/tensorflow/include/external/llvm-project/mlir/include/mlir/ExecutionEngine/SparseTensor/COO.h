//===- COO.h - Coordinate-scheme sparse tensor representation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a coordinate-scheme representation of sparse tensors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H
#define MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <functional>
#include <vector>

namespace mlir {
namespace sparse_tensor {

/// An element of a sparse tensor in coordinate-scheme representation
/// (i.e., a pair of coordinates and value). For example, a rank-1
/// vector element would look like
///   ({i}, a[i])
/// and a rank-5 tensor element would look like
///   ({i,j,k,l,m}, a[i,j,k,l,m])
///
/// The coordinates are represented as a (non-owning) pointer into a
/// shared pool of coordinates, rather than being stored directly in this
/// object. This significantly improves performance because it reduces the
/// per-element memory footprint and centralizes the memory management for
/// coordinates. The only downside is that the coordinates themselves cannot
/// be retrieved without knowing the rank of the tensor to which this element
/// belongs (and that rank is not stored in this object).
template <typename V>
struct Element final {
  Element(const uint64_t *coords, V val) : coords(coords), value(val){};
  const uint64_t *coords; // pointer into shared coordinates pool
  V value;
};

/// Closure object for `operator<` on `Element` with a given rank.
template <typename V>
struct ElementLT final {
  ElementLT(uint64_t rank) : rank(rank) {}
  bool operator()(const Element<V> &e1, const Element<V> &e2) const {
    for (uint64_t d = 0; d < rank; ++d) {
      if (e1.coords[d] == e2.coords[d])
        continue;
      return e1.coords[d] < e2.coords[d];
    }
    return false;
  }
  const uint64_t rank;
};

/// A memory-resident sparse tensor in coordinate-scheme representation
/// (a collection of `Element`s). This data structure is used as an
/// intermediate representation, e.g., for reading sparse tensors from
/// external formats into memory.
template <typename V>
class SparseTensorCOO final {
public:
  /// Constructs a new coordinate-scheme sparse tensor with the given
  /// sizes and an optional initial storage capacity.
  explicit SparseTensorCOO(const std::vector<uint64_t> &dimSizes,
                           uint64_t capacity = 0)
      : SparseTensorCOO(dimSizes.size(), dimSizes.data(), capacity) {}

  /// Constructs a new coordinate-scheme sparse tensor with the given
  /// sizes and an optional initial storage capacity. The size of the
  /// dimSizes array is determined by dimRank.
  explicit SparseTensorCOO(uint64_t dimRank, const uint64_t *dimSizes,
                           uint64_t capacity = 0)
      : dimSizes(dimSizes, dimSizes + dimRank), isSorted(true) {
    assert(dimRank > 0 && "Trivial shape is not supported");
    for (uint64_t d = 0; d < dimRank; ++d)
      assert(dimSizes[d] > 0 && "Dimension size zero has trivial storage");
    if (capacity) {
      elements.reserve(capacity);
      coordinates.reserve(capacity * dimRank);
    }
  }

  /// Gets the dimension-rank of the tensor.
  uint64_t getRank() const { return dimSizes.size(); }

  /// Gets the dimension-sizes array.
  const std::vector<uint64_t> &getDimSizes() const { return dimSizes; }

  /// Gets the elements array.
  const std::vector<Element<V>> &getElements() const { return elements; }

  /// Returns the `operator<` closure object for the COO's element type.
  ElementLT<V> getElementLT() const { return ElementLT<V>(getRank()); }

  /// Adds an element to the tensor.
  void add(const std::vector<uint64_t> &dimCoords, V val) {
    const uint64_t *base = coordinates.data();
    const uint64_t size = coordinates.size();
    const uint64_t dimRank = getRank();
    assert(dimCoords.size() == dimRank && "Element rank mismatch");
    for (uint64_t d = 0; d < dimRank; ++d) {
      assert(dimCoords[d] < dimSizes[d] &&
             "Coordinate is too large for the dimension");
      coordinates.push_back(dimCoords[d]);
    }
    // This base only changes if `coordinates` was reallocated. In which
    // case, we need to correct all previous pointers into the vector.
    // Note that this only happens if we did not set the initial capacity
    // right, and then only for every internal vector reallocation (which
    // with the doubling rule should only incur an amortized linear overhead).
    const uint64_t *const newBase = coordinates.data();
    if (newBase != base) {
      for (uint64_t i = 0, n = elements.size(); i < n; ++i)
        elements[i].coords = newBase + (elements[i].coords - base);
      base = newBase;
    }
    // Add the new element and update the sorted bit.
    const Element<V> addedElem(base + size, val);
    if (!elements.empty() && isSorted)
      isSorted = getElementLT()(elements.back(), addedElem);
    elements.push_back(addedElem);
  }

  /// Sorts elements lexicographically by coordinates. If a coordinate
  /// is mapped to multiple values, then the relative order of those
  /// values is unspecified.
  void sort() {
    if (isSorted)
      return;
    std::sort(elements.begin(), elements.end(), getElementLT());
    isSorted = true;
  }

private:
  const std::vector<uint64_t> dimSizes; // per-dimension sizes
  std::vector<Element<V>> elements;     // all COO elements
  std::vector<uint64_t> coordinates;    // shared coordinate pool
  bool isSorted;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_SPARSETENSOR_COO_H
