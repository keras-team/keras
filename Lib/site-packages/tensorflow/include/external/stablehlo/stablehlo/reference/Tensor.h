/* Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_REFERENCE_TENSOR_H
#define STABLEHLO_REFERENCE_TENSOR_H

#include <numeric>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "stablehlo/reference/Axes.h"
#include "stablehlo/reference/Element.h"
#include "stablehlo/reference/Index.h"

namespace mlir {
namespace stablehlo {

namespace detail {

/// Underlying storage class for Tensor objects.
class Buffer : public llvm::ThreadSafeRefCountedBase<Buffer> {
 public:
  /// \name Constructors
  /// @{
  explicit Buffer(ShapedType type);
  Buffer(ShapedType type, AsmResourceBlob blob);
  Buffer(Buffer &&other) = default;
  /// @}

  /// Move assignment operator deleted in RefCountedBase
  Buffer &operator=(Buffer &&other) = delete;

  /// Returns type of the Buffer object.
  ShapedType getType() { return type_; }

  /// Provides access to the underlying non-mutable storage.
  ArrayRef<char> getData() const { return blob_.getData(); }

  /// Provides access to the underlying mutable storage.
  MutableArrayRef<char> getMutableData() { return blob_.getMutableData(); }

 private:
  ShapedType type_;
  AsmResourceBlob blob_;
};

}  // namespace detail

/// Class to model a tensor, an n-dimensional array. Provide access to
/// individual elements of the tensor using n-dimensional indices.
class Tensor {
 public:
  /// \name Constructors
  /// @{
  Tensor();
  explicit Tensor(ShapedType type);
  explicit Tensor(ShapedType type, AsmResourceBlob blob);
  Tensor(const Tensor &other) = default;
  /// @}

  /// Assignment operator.
  Tensor &operator=(const Tensor &other) = default;

  /// Boolean conversion operator.
  explicit operator bool() const { return (bool)impl_; }

  /// Logical not operator.
  bool operator!() const { return !impl_; }

  /// Returns type of the Tensor object.
  ShapedType getType() const { return impl_->getType(); };

  /// Returns rank of the Tensor object.
  int64_t getRank() const { return impl_->getType().getRank(); }

  /// Returns axes of the Tensor object: [0, 1, ..., getRank() - 1].
  Axes getAxes() const {
    Axes result(getRank());
    std::iota(result.begin(), result.end(), 0);
    return result;
  }

  /// Returns shape of the Tensor object.
  Sizes getShape() const { return Sizes(impl_->getType().getShape()); }

  /// Returns the number of elements.
  int64_t getNumElements() const { return impl_->getType().getNumElements(); }

  /// Returns element type of the Tensor object.
  Type getElementType() const { return impl_->getType().getElementType(); };

  /// Provides read access to the tensor element indexed at 'index'.
  Element get(const Index &index) const;

  /// Provides read access to underlying tensor data buffer.
  const char *getData() const { return impl_->getData().data(); }

  /// Provides write access to the tensor element indexed at 'index'.
  ///
  /// \param index The multi-dimensional index to write to.
  /// \param element The Element object \a element is used to update the
  /// underlying storage pointed to by \a index.
  void set(const Index &index, const Element &element);

  /// Prints Tensor objects.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Iterate over the index space of a Tensor object.
  IndexSpaceIterator index_begin() const;
  IndexSpaceIterator index_end() const;

 private:
  llvm::IntrusiveRefCntPtr<detail::Buffer> impl_;
};

/// Print utilities for Tensor objects.
inline raw_ostream &operator<<(raw_ostream &os, Tensor tensor) {
  tensor.print(os);
  return os;
}

/// Creates a Tensor from a DenseElementsAttr.
Tensor makeTensor(DenseElementsAttr attr);

/// Creates a DenseElementsAttr from a Tensor.
DenseElementsAttr makeDenseElementsAttr(Tensor tensor);

/// Creates a Sizes from a Tensor.
Sizes makeSizes(Tensor tensor);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_TENSOR_H
