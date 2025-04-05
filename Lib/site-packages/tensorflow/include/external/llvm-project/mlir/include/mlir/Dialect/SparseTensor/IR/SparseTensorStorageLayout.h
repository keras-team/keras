//===- SparseTensorStorageLayout.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utilities for the sparse memory layout.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORSTORAGELAYOUT_H_
#define MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORSTORAGELAYOUT_H_

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"

namespace mlir {
namespace sparse_tensor {

///===----------------------------------------------------------------------===//
/// The sparse tensor storage scheme for a tensor is organized as a single
/// compound type with the following fields. Note that every memref with `?`
/// size actually behaves as a "vector", i.e. the stored size is the capacity
/// and the used size resides in the storage_specifier struct.
///
/// struct {
///   ; per-level l:
///   ;  if dense:
///        <nothing>
///   ;  if compressed:
///        memref<[batch] x ? x pos>  positions   ; positions for level l
///        memref<[batch] x ? x crd>  coordinates ; coordinates for level l
///   ;  if loose-[batch] x compressed:
///        memref<[batch] x ? x pos>  positions   ; lo/hi pos pairs for level l
///        memref<[batch] x ? x crd>  coordinates ; coordinates for level l
///   ;  if singleton/2-out-of-4:
///        memref<[batch] x ? x crd>  coordinates ; coordinates for level l
///
///   memref<[batch] x ? x eltType> values        ; values
///
///   struct sparse_tensor.storage_specifier {
///     array<rank x int> lvlSizes    ; sizes/cardinalities for each level
///     // TODO: memSizes need to be expanded to array<[batch] x n x int> to
///     // support different sizes for different batches. At the moment, we
///     // assume that every batch occupies the same memory size.
///     array<n x int> memSizes       ; sizes/lengths for each data memref
///   }
/// };
///
/// In addition, for a "trailing COO region", defined as a compressed level
/// followed by one or more singleton levels, the default SOA storage that
/// is inherent to the TACO format is optimized into an AOS storage where
/// all coordinates of a stored element appear consecutively.  In such cases,
/// a special operation (sparse_tensor.coordinates_buffer) must be used to
/// access the AOS coordinates array. In the code below, the method
/// `getCOOStart` is used to find the start of the "trailing COO region".
///
/// If the sparse tensor is a slice (produced by `tensor.extract_slice`
/// operation), instead of allocating a new sparse tensor for it, it reuses the
/// same sets of MemRefs but attaching a additional set of slicing-metadata for
/// per-dimension slice offset and stride.
///
/// Examples.
///
/// #CSR storage of 2-dim matrix yields
///  memref<?xindex>                           ; positions-1
///  memref<?xindex>                           ; coordinates-1
///  memref<?xf64>                             ; values
///  struct<(array<2 x i64>, array<3 x i64>)>) ; lvl0, lvl1, 3xsizes
///
/// #COO storage of 2-dim matrix yields
///  memref<?xindex>,                          ; positions-0, essentially [0,sz]
///  memref<?xindex>                           ; AOS coordinates storage
///  memref<?xf64>                             ; values
///  struct<(array<2 x i64>, array<3 x i64>)>) ; lvl0, lvl1, 3xsizes
///
/// Slice on #COO storage of 2-dim matrix yields
///  ;; Inherited from the original sparse tensors
///  memref<?xindex>,                          ; positions-0, essentially [0,sz]
///  memref<?xindex>                           ; AOS coordinates storage
///  memref<?xf64>                             ; values
///  struct<(array<2 x i64>, array<3 x i64>,   ; lvl0, lvl1, 3xsizes
///  ;; Extra slicing-metadata
///          array<2 x i64>, array<2 x i64>)>) ; dim offset, dim stride.
///
///===----------------------------------------------------------------------===//

enum class SparseTensorFieldKind : uint32_t {
  StorageSpec = 0,
  PosMemRef = static_cast<uint32_t>(StorageSpecifierKind::PosMemSize),
  CrdMemRef = static_cast<uint32_t>(StorageSpecifierKind::CrdMemSize),
  ValMemRef = static_cast<uint32_t>(StorageSpecifierKind::ValMemSize)
};

inline StorageSpecifierKind toSpecifierKind(SparseTensorFieldKind kind) {
  assert(kind != SparseTensorFieldKind::StorageSpec);
  return static_cast<StorageSpecifierKind>(kind);
}

inline SparseTensorFieldKind toFieldKind(StorageSpecifierKind kind) {
  assert(kind != StorageSpecifierKind::LvlSize);
  return static_cast<SparseTensorFieldKind>(kind);
}

/// The type of field indices.  This alias is to help code be more
/// self-documenting; unfortunately it is not type-checked, so it only
/// provides documentation rather than doing anything to prevent mixups.
using FieldIndex = unsigned;

/// Provides methods to access fields of a sparse tensor with the given
/// encoding.
class StorageLayout {
public:
  explicit StorageLayout(const SparseTensorType &stt)
      : StorageLayout(stt.getEncoding()) {}
  explicit StorageLayout(SparseTensorEncodingAttr enc) : enc(enc) {
    assert(enc);
  }

  /// For each field that will be allocated for the given sparse tensor
  /// encoding, calls the callback with the corresponding field index,
  /// field kind, level, and level-type (the last two are only for level
  /// memrefs).  The field index always starts with zero and increments
  /// by one between each callback invocation.  Ideally, all other methods
  /// should rely on this function to query a sparse tensor fields instead
  /// of relying on ad-hoc index computation.
  void foreachField(
      llvm::function_ref<bool(
          FieldIndex /*fieldIdx*/, SparseTensorFieldKind /*fieldKind*/,
          Level /*lvl (if applicable)*/, LevelType /*LT (if applicable)*/)>)
      const;

  /// Gets the field index for required field.
  FieldIndex getMemRefFieldIndex(SparseTensorFieldKind kind,
                                 std::optional<Level> lvl) const {
    return getFieldIndexAndStride(kind, lvl).first;
  }

  /// Gets the total number of fields for the given sparse tensor encoding.
  unsigned getNumFields() const;

  /// Gets the total number of data fields (coordinate arrays, position
  /// arrays, and a value array) for the given sparse tensor encoding.
  unsigned getNumDataFields() const;

  std::pair<FieldIndex, unsigned>
  getFieldIndexAndStride(SparseTensorFieldKind kind,
                         std::optional<Level> lvl) const;

private:
  const SparseTensorEncodingAttr enc;
};

//
// Wrapper functions to invoke StorageLayout-related method.
//

inline unsigned getNumFieldsFromEncoding(SparseTensorEncodingAttr enc) {
  return StorageLayout(enc).getNumFields();
}

inline unsigned getNumDataFieldsFromEncoding(SparseTensorEncodingAttr enc) {
  return StorageLayout(enc).getNumDataFields();
}

inline void foreachFieldInSparseTensor(
    SparseTensorEncodingAttr enc,
    llvm::function_ref<bool(FieldIndex, SparseTensorFieldKind, Level,
                            LevelType)>
        callback) {
  return StorageLayout(enc).foreachField(callback);
}

void foreachFieldAndTypeInSparseTensor(
    SparseTensorType,
    llvm::function_ref<bool(Type, FieldIndex, SparseTensorFieldKind, Level,
                            LevelType)>);

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_IR_SPARSETENSORSTORAGELAYOUT_H_
