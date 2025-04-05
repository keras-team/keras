//===- RaggedArray.h - 2D array with different inner lengths ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <iterator>

namespace mlir {
/// A 2D array where each row may have different length. Elements of each row
/// are stored contiguously, but rows don't have a fixed order in the storage.
template <typename T>
class RaggedArray {
public:
  /// Returns the number of rows in the 2D array.
  size_t size() const { return slices.size(); }

  /// Returns true if the are no rows in the 2D array. Note that an array with a
  /// non-zero number of empty rows is *NOT* empty.
  bool empty() const { return slices.empty(); }

  /// Accesses `pos`-th row.
  ArrayRef<T> operator[](size_t pos) const { return at(pos); }
  ArrayRef<T> at(size_t pos) const {
    if (slices[pos].first == static_cast<size_t>(-1))
      return ArrayRef<T>();
    return ArrayRef<T>(storage).slice(slices[pos].first, slices[pos].second);
  }
  MutableArrayRef<T> operator[](size_t pos) { return at(pos); }
  MutableArrayRef<T> at(size_t pos) {
    if (slices[pos].first == static_cast<size_t>(-1))
      return MutableArrayRef<T>();
    return MutableArrayRef<T>(storage).slice(slices[pos].first,
                                             slices[pos].second);
  }

  /// Iterator over the rows.
  class iterator
      : public llvm::iterator_facade_base<
            iterator, std::forward_iterator_tag, MutableArrayRef<T>,
            std::ptrdiff_t, MutableArrayRef<T> *, MutableArrayRef<T>> {
  public:
    /// Creates the start iterator.
    explicit iterator(RaggedArray &ragged) : ragged(ragged), pos(0) {}

    /// Creates the end iterator.
    iterator(RaggedArray &ragged, size_t pos) : ragged(ragged), pos(pos) {}

    /// Dereferences the current iterator. Assumes in-bounds.
    MutableArrayRef<T> operator*() const { return ragged[pos]; }

    /// Increments the iterator.
    iterator &operator++() {
      if (pos < ragged.slices.size())
        ++pos;
      return *this;
    }

    /// Compares the two iterators. Iterators into different ragged arrays
    /// compare not equal.
    bool operator==(const iterator &other) const {
      return &ragged == &other.ragged && pos == other.pos;
    }

  private:
    RaggedArray &ragged;
    size_t pos;
  };

  /// Constant iterator over the rows.
  class const_iterator
      : public llvm::iterator_facade_base<
            const_iterator, std::forward_iterator_tag, ArrayRef<T>,
            std::ptrdiff_t, ArrayRef<T> *, ArrayRef<T>> {
  public:
    /// Creates the start iterator.
    explicit const_iterator(const RaggedArray &ragged)
        : ragged(ragged), pos(0) {}

    /// Creates the end iterator.
    const_iterator(const RaggedArray &ragged, size_t pos)
        : ragged(ragged), pos(pos) {}

    /// Dereferences the current iterator. Assumes in-bounds.
    ArrayRef<T> operator*() const { return ragged[pos]; }

    /// Increments the iterator.
    const_iterator &operator++() {
      if (pos < ragged.slices.size())
        ++pos;
      return *this;
    }

    /// Compares the two iterators. Iterators into different ragged arrays
    /// compare not equal.
    bool operator==(const const_iterator &other) const {
      return &ragged == &other.ragged && pos == other.pos;
    }

  private:
    const RaggedArray &ragged;
    size_t pos;
  };

  /// Iterator over rows.
  const_iterator begin() const { return const_iterator(*this); }
  const_iterator end() const { return const_iterator(*this, slices.size()); }
  iterator begin() { return iterator(*this); }
  iterator end() { return iterator(*this, slices.size()); }

  /// Reserve space to store `size` rows with `nestedSize` elements each.
  void reserve(size_t size, size_t nestedSize = 0) {
    slices.reserve(size);
    storage.reserve(size * nestedSize);
  }

  /// Appends the given range of elements as a new row to the 2D array. May
  /// invalidate the end iterator.
  template <typename Range>
  void push_back(Range &&elements) {
    slices.push_back(appendToStorage(std::forward<Range>(elements)));
  }

  /// Replaces the `pos`-th row in the 2D array with the given range of
  /// elements. Invalidates iterators and references to `pos`-th and all
  /// succeeding rows.
  template <typename Range>
  void replace(size_t pos, Range &&elements) {
    if (slices[pos].first != static_cast<size_t>(-1)) {
      auto from = std::next(storage.begin(), slices[pos].first);
      auto to = std::next(from, slices[pos].second);
      auto newFrom = storage.erase(from, to);
      // Update the array refs after the underlying storage was shifted.
      for (size_t i = pos + 1, e = size(); i < e; ++i) {
        slices[i] = std::make_pair(std::distance(storage.begin(), newFrom),
                                   slices[i].second);
        std::advance(newFrom, slices[i].second);
      }
    }
    slices[pos] = appendToStorage(std::forward<Range>(elements));
  }

  /// Appends `num` empty rows to the array.
  void appendEmptyRows(size_t num) {
    slices.resize(slices.size() + num, std::pair<size_t, size_t>(-1, 0));
  }

  /// Removes the first subarray in-place. Invalidates iterators to all rows.
  void removeFront() { slices.erase(slices.begin()); }

private:
  /// Appends the given elements to the storage and returns an ArrayRef
  /// pointing to them in the storage.
  template <typename Range>
  std::pair<size_t, size_t> appendToStorage(Range &&elements) {
    size_t start = storage.size();
    llvm::append_range(storage, std::forward<Range>(elements));
    return std::make_pair(start, storage.size() - start);
  }

  /// Outer elements of the ragged array. Each entry is an (offset, length)
  /// pair identifying a contiguous segment in the `storage` list that
  /// contains the actual elements. This allows for elements to be stored
  /// contiguously without nested vectors and for different segments to be set
  /// or replaced in any order.
  SmallVector<std::pair<size_t, size_t>> slices;

  /// Dense storage for ragged array elements.
  SmallVector<T> storage;
};
} // namespace mlir
