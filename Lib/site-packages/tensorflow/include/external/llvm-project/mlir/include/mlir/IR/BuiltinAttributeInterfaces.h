//===- BuiltinAttributeInterfaces.h - Builtin Attr Interfaces ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINATTRIBUTEINTERFACES_H
#define MLIR_IR_BUILTINATTRIBUTEINTERFACES_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"
#include "llvm/Support/raw_ostream.h"
#include <complex>
#include <optional>

namespace mlir {

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//
namespace detail {
/// This class provides support for indexing into the element range of an
/// ElementsAttr. It is used to opaquely wrap either a contiguous range, via
/// `ElementsAttrIndexer::contiguous`, or a non-contiguous range, via
/// `ElementsAttrIndexer::nonContiguous`, A contiguous range is an array-like
/// range, where all of the elements are layed out sequentially in memory. A
/// non-contiguous range implies no contiguity, and elements may even be
/// materialized when indexing, such as the case for a mapped_range.
struct ElementsAttrIndexer {
public:
  ElementsAttrIndexer()
      : ElementsAttrIndexer(/*isContiguous=*/true, /*isSplat=*/true) {}
  ElementsAttrIndexer(ElementsAttrIndexer &&rhs)
      : isContiguous(rhs.isContiguous), isSplat(rhs.isSplat) {
    if (isContiguous)
      conState = rhs.conState;
    else
      new (&nonConState) NonContiguousState(std::move(rhs.nonConState));
  }
  ElementsAttrIndexer(const ElementsAttrIndexer &rhs)
      : isContiguous(rhs.isContiguous), isSplat(rhs.isSplat) {
    if (isContiguous)
      conState = rhs.conState;
    else
      new (&nonConState) NonContiguousState(rhs.nonConState);
  }
  ~ElementsAttrIndexer() {
    if (!isContiguous)
      nonConState.~NonContiguousState();
  }

  /// Construct an indexer for a non-contiguous range starting at the given
  /// iterator. A non-contiguous range implies no contiguity, and elements may
  /// even be materialized when indexing, such as the case for a mapped_range.
  template <typename IteratorT>
  static ElementsAttrIndexer nonContiguous(bool isSplat, IteratorT &&iterator) {
    ElementsAttrIndexer indexer(/*isContiguous=*/false, isSplat);
    new (&indexer.nonConState)
        NonContiguousState(std::forward<IteratorT>(iterator));
    return indexer;
  }

  // Construct an indexer for a contiguous range starting at the given element
  // pointer. A contiguous range is an array-like range, where all of the
  // elements are layed out sequentially in memory.
  template <typename T>
  static ElementsAttrIndexer contiguous(bool isSplat, const T *firstEltPtr) {
    ElementsAttrIndexer indexer(/*isContiguous=*/true, isSplat);
    new (&indexer.conState) ContiguousState(firstEltPtr);
    return indexer;
  }

  /// Access the element at the given index.
  template <typename T>
  T at(uint64_t index) const {
    if (isSplat)
      index = 0;
    return isContiguous ? conState.at<T>(index) : nonConState.at<T>(index);
  }

private:
  ElementsAttrIndexer(bool isContiguous, bool isSplat)
      : isContiguous(isContiguous), isSplat(isSplat), conState(nullptr) {}

  /// This class contains all of the state necessary to index a contiguous
  /// range.
  class ContiguousState {
  public:
    ContiguousState(const void *firstEltPtr) : firstEltPtr(firstEltPtr) {}

    /// Access the element at the given index.
    template <typename T>
    const T &at(uint64_t index) const {
      return *(reinterpret_cast<const T *>(firstEltPtr) + index);
    }

  private:
    const void *firstEltPtr;
  };

  /// This class contains all of the state necessary to index a non-contiguous
  /// range.
  class NonContiguousState {
  private:
    /// This class is used to represent the abstract base of an opaque iterator.
    /// This allows for all iterator and element types to be completely
    /// type-erased.
    struct OpaqueIteratorBase {
      virtual ~OpaqueIteratorBase() = default;
      virtual std::unique_ptr<OpaqueIteratorBase> clone() const = 0;
    };
    /// This class is used to represent the abstract base of an opaque iterator
    /// that iterates over elements of type `T`. This allows for all iterator
    /// types to be completely type-erased.
    template <typename T>
    struct OpaqueIteratorValueBase : public OpaqueIteratorBase {
      virtual T at(uint64_t index) = 0;
    };
    /// This class is used to represent an opaque handle to an iterator of type
    /// `IteratorT` that iterates over elements of type `T`.
    template <typename IteratorT, typename T>
    struct OpaqueIterator : public OpaqueIteratorValueBase<T> {
      template <typename ItTy, typename FuncTy, typename FuncReturnTy>
      static void isMappedIteratorTestFn(
          llvm::mapped_iterator<ItTy, FuncTy, FuncReturnTy>) {}
      template <typename U, typename... Args>
      using is_mapped_iterator =
          decltype(isMappedIteratorTestFn(std::declval<U>()));
      template <typename U>
      using detect_is_mapped_iterator =
          llvm::is_detected<is_mapped_iterator, U>;

      /// Access the element within the iterator at the given index.
      template <typename ItT>
      static std::enable_if_t<!detect_is_mapped_iterator<ItT>::value, T>
      atImpl(ItT &&it, uint64_t index) {
        return *std::next(it, index);
      }
      template <typename ItT>
      static std::enable_if_t<detect_is_mapped_iterator<ItT>::value, T>
      atImpl(ItT &&it, uint64_t index) {
        // Special case mapped_iterator to avoid copying the function.
        return it.getFunction()(*std::next(it.getCurrent(), index));
      }

    public:
      template <typename U>
      OpaqueIterator(U &&iterator) : iterator(std::forward<U>(iterator)) {}
      std::unique_ptr<OpaqueIteratorBase> clone() const final {
        return std::make_unique<OpaqueIterator<IteratorT, T>>(iterator);
      }

      /// Access the element at the given index.
      T at(uint64_t index) final { return atImpl(iterator, index); }

    private:
      IteratorT iterator;
    };

  public:
    /// Construct the state with the given iterator type.
    template <typename IteratorT, typename T = typename llvm::remove_cvref_t<
                                      decltype(*std::declval<IteratorT>())>>
    NonContiguousState(IteratorT iterator)
        : iterator(std::make_unique<OpaqueIterator<IteratorT, T>>(iterator)) {}
    NonContiguousState(const NonContiguousState &other)
        : iterator(other.iterator->clone()) {}
    NonContiguousState(NonContiguousState &&other) = default;

    /// Access the element at the given index.
    template <typename T>
    T at(uint64_t index) const {
      auto *valueIt = static_cast<OpaqueIteratorValueBase<T> *>(iterator.get());
      return valueIt->at(index);
    }

    /// The opaque iterator state.
    std::unique_ptr<OpaqueIteratorBase> iterator;
  };

  /// A boolean indicating if this range is contiguous or not.
  bool isContiguous;
  /// A boolean indicating if this range is a splat.
  bool isSplat;
  /// The underlying range state.
  union {
    ContiguousState conState;
    NonContiguousState nonConState;
  };
};

/// This class implements a generic iterator for ElementsAttr.
template <typename T>
class ElementsAttrIterator
    : public llvm::iterator_facade_base<ElementsAttrIterator<T>,
                                        std::random_access_iterator_tag, T,
                                        std::ptrdiff_t, T, T> {
public:
  ElementsAttrIterator(ElementsAttrIndexer indexer, size_t dataIndex)
      : indexer(std::move(indexer)), index(dataIndex) {}

  // Boilerplate iterator methods.
  ptrdiff_t operator-(const ElementsAttrIterator &rhs) const {
    return index - rhs.index;
  }
  bool operator==(const ElementsAttrIterator &rhs) const {
    return index == rhs.index;
  }
  bool operator<(const ElementsAttrIterator &rhs) const {
    return index < rhs.index;
  }
  ElementsAttrIterator &operator+=(ptrdiff_t offset) {
    index += offset;
    return *this;
  }
  ElementsAttrIterator &operator-=(ptrdiff_t offset) {
    index -= offset;
    return *this;
  }

  /// Return the value at the current iterator position.
  T operator*() const { return indexer.at<T>(index); }

private:
  ElementsAttrIndexer indexer;
  ptrdiff_t index;
};

/// This class provides iterator utilities for an ElementsAttr range.
template <typename IteratorT>
class ElementsAttrRange : public llvm::iterator_range<IteratorT> {
public:
  using reference = typename IteratorT::reference;

  ElementsAttrRange(ShapedType shapeType,
                    const llvm::iterator_range<IteratorT> &range)
      : llvm::iterator_range<IteratorT>(range), shapeType(shapeType) {}
  ElementsAttrRange(ShapedType shapeType, IteratorT beginIt, IteratorT endIt)
      : ElementsAttrRange(shapeType, llvm::make_range(beginIt, endIt)) {}

  /// Return the value at the given index.
  reference operator[](ArrayRef<uint64_t> index) const;
  reference operator[](uint64_t index) const {
    return *std::next(this->begin(), index);
  }

  /// Return the size of this range.
  size_t size() const { return llvm::size(*this); }

private:
  /// The shaped type of the parent ElementsAttr.
  ShapedType shapeType;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// MemRefLayoutAttrInterface
//===----------------------------------------------------------------------===//

namespace detail {

// Verify the affine map 'm' can be used as a layout specification
// for memref with 'shape'.
LogicalResult
verifyAffineMapAsLayout(AffineMap m, ArrayRef<int64_t> shape,
                        function_ref<InFlightDiagnostic()> emitError);

} // namespace detail

} // namespace mlir

//===----------------------------------------------------------------------===//
// Tablegen Interface Declarations
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributeInterfaces.h.inc"

//===----------------------------------------------------------------------===//
// ElementsAttr
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
/// Return the value at the given index.
template <typename IteratorT>
auto ElementsAttrRange<IteratorT>::operator[](ArrayRef<uint64_t> index) const
    -> reference {
  // Skip to the element corresponding to the flattened index.
  return (*this)[ElementsAttr::getFlattenedIndex(shapeType, index)];
}
} // namespace detail

/// Return the elements of this attribute as a value of type 'T'.
template <typename T>
auto ElementsAttr::value_begin() const -> DefaultValueCheckT<T, iterator<T>> {
  if (std::optional<iterator<T>> iterator = try_value_begin<T>())
    return std::move(*iterator);
  llvm::errs()
      << "ElementsAttr does not provide iteration facilities for type `"
      << llvm::getTypeName<T>() << "`, see attribute: " << *this << "\n";
  llvm_unreachable("invalid `T` for ElementsAttr::getValues");
}
template <typename T>
auto ElementsAttr::try_value_begin() const
    -> DefaultValueCheckT<T, std::optional<iterator<T>>> {
  FailureOr<detail::ElementsAttrIndexer> indexer =
      getValuesImpl(TypeID::get<T>());
  if (failed(indexer))
    return std::nullopt;
  return iterator<T>(std::move(*indexer), 0);
}
} // namespace mlir.

#endif // MLIR_IR_BUILTINATTRIBUTEINTERFACES_H
