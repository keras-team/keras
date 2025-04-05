//===- GPUDialect.h - MLIR Dialect for GPU Kernels --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_IR_GPUDIALECT_H
#define MLIR_DIALECT_GPU_IR_GPUDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/DLTI/Traits.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace gpu {

/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  Value x;
  Value y;
  Value z;
};

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;

  static constexpr StringLiteral name = "gpu.async_token";
};

/// MMAMatrixType storage and uniquing. Array is uniqued based on its shape
/// and type.
struct MMAMatrixStorageType : public TypeStorage {
  MMAMatrixStorageType(unsigned numDims, const int64_t *dimShapes,
                       Type elementType, StringRef operand)
      : dimShapes(dimShapes), numDims(numDims), elementType(elementType),
        operand(operand) {}

  /// The hash key for uniquing.
  using KeyTy = std::tuple<ArrayRef<int64_t>, Type, StringRef>;
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(getShape(), elementType, operand);
  }

  /// Construction.
  static MMAMatrixStorageType *construct(TypeStorageAllocator &allocator,
                                         const KeyTy &key) {
    ArrayRef<int64_t> shape = allocator.copyInto(std::get<0>(key));
    StringRef operand = allocator.copyInto(std::get<2>(key));

    return new (allocator.allocate<MMAMatrixStorageType>())
        MMAMatrixStorageType(shape.size(), shape.data(), std::get<1>(key),
                             operand);
  }

  ArrayRef<int64_t> getShape() const {
    return ArrayRef<int64_t>(dimShapes, numDims);
  }

  StringRef getOperand() const { return operand; }

  /// Reference to the shape of the MMA matrix.
  const int64_t *dimShapes;

  /// Number of dimensions in the MMA matrix.
  unsigned numDims;

  /// Element type of elements held in the MMA matrix.
  Type elementType;

  /// MMA operand that this MMAMatrix holds. The general form of operation this
  /// type supports is given by the equation C += A*B. This field specifies
  /// which operand in the given equation is held by this type. The valid values
  /// are "AOp", "BOp" and "COp".
  StringRef operand;
};

/// MMAMatrix represents a matrix held by a subgroup for matrix-matrix multiply
/// accumulate operations. MMAMatrices are taken as direct operands by these
/// operations and are also produced as results. These matrices are meant to
/// reside in the registers. A limited number of pointwise operations can be
/// performed on these matrices, i.e., operations which operate uniformly on
/// all the elements in the matrix and do not change the order of matrix
/// elements. The above conditions exist because the layout of matrix elements
/// inside the matrix is opaque i.e., the elements may be present in the
/// matrix in any order. The general usage of this type is shown as follows:-
///
///   %0 = gpu.subgroup_mma_load_matrix %arg0[%c0, %c0] {leadDimension = 16 :
///           index} : memref<16x16xf16> -> !gpu.mma_matrix<16x16xf16, "AOp">
///
/// The MMAMatrixType describes the shape of the matrix being loaded and the
/// operand being loaded too. The operand needs to be specified to aid the
/// lowering of this type to dialects such as NVVM where each workitem may
/// hold different amount of elements depending on the elementType of the
/// matrix. For e.g., Each workitem holds 4 vector<2xf16>s for f16 data type
/// and 8 f32s for f32 data type of MMAMatrix. Some other instances of usage
/// are:-
///
///   %3 = gpu.subgroup_mma_compute %0, %1, %2 :
///   !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">
///    -> !gpu.mma_matrix<16x16xf32, "COp">
///
///
///   gpu.subgroup_mma_store_matrix %3, %arg22[%c0, %c0] {leadDimension = 16
///           : index}: !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32>
// TODO: consider moving this to ODS.
class MMAMatrixType
    : public Type::TypeBase<MMAMatrixType, Type, MMAMatrixStorageType> {
public:
  using Base::Base;

  static constexpr StringLiteral name = "gpu.mma_matrix";

  /// Get MMAMatrixType and verify construction Invariants.
  static MMAMatrixType get(ArrayRef<int64_t> shape, Type elementType,
                           StringRef operand);

  /// Get MMAMatrixType at a particular location and verify construction
  /// Invariants.
  static MMAMatrixType getChecked(function_ref<InFlightDiagnostic()> emitError,
                                  ArrayRef<int64_t> shape, Type elementType,
                                  StringRef operand);

  /// Check if a type is valid a MMAMatrixType elementType.
  static bool isValidElementType(Type elementType);

  /// Verify that shape and elementType are actually allowed for the
  /// MMAMatrixType.
  static LogicalResult
  verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<int64_t> shape, Type elementType,
                   StringRef operand);

  /// Get number of dims.
  unsigned getNumDims() const;

  /// Get shape of the matrix.
  ArrayRef<int64_t> getShape() const;

  /// Get elementType of a single element.
  Type getElementType() const;

  /// The general form of operation this type supports is given by the equation
  /// C += A*B. This function returns which operand in the given equation is
  /// held by this type. String returned can be one of"AOp", "BOp" and "COp".
  StringRef getOperand() const;
};

// Adds a `gpu.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

// Handle types for sparse.
enum class SparseHandleKind { SpMat, DnTensor, SpGEMMOp };

class SparseDnTensorHandleType
    : public Type::TypeBase<SparseDnTensorHandleType, Type, TypeStorage> {
public:
  using Base = typename Type::TypeBase<SparseDnTensorHandleType, Type,
                                       TypeStorage>::Base;
  using Base::Base;

  static constexpr StringLiteral name = "gpu.sparse.dntensor_handle";
};

class SparseSpMatHandleType
    : public Type::TypeBase<SparseSpMatHandleType, Type, TypeStorage> {
public:
  using Base =
      typename Type::TypeBase<SparseSpMatHandleType, Type, TypeStorage>::Base;
  using Base::Base;

  static constexpr StringLiteral name = "gpu.sparse.spmat_handle";
};

class SparseSpGEMMOpHandleType
    : public Type::TypeBase<SparseSpGEMMOpHandleType, Type, TypeStorage> {
public:
  using Base = typename Type::TypeBase<SparseSpGEMMOpHandleType, Type,
                                       TypeStorage>::Base;
  using Base::Base;

  static constexpr StringLiteral name = "gpu.sparse.spgemmop_handle";
};

} // namespace gpu
} // namespace mlir

#include "mlir/Dialect/GPU/IR/GPUOpsEnums.h.inc"

#include "mlir/Dialect/GPU/IR/GPUOpsDialect.h.inc"

#include "mlir/Dialect/GPU/IR/GPUOpInterfaces.h.inc"

#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOps.h.inc"

#endif // MLIR_DIALECT_GPU_IR_GPUDIALECT_H
