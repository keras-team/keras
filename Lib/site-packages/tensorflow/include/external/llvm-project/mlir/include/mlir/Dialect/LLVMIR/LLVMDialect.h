//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

namespace llvm {
class Type;
class LLVMContext;
namespace sys {
template <bool mt_only>
class SmartMutex;
} // namespace sys
} // namespace llvm

namespace mlir {
namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMTypeStorage;
struct LLVMDialectImpl;
} // namespace detail
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace LLVM {
template <typename Values>
class GEPIndicesAdaptor;

/// Bit-width of a 'GEPConstantIndex' within GEPArg.
constexpr int kGEPConstantBitWidth = 29;
/// Wrapper around a int32_t for use in a PointerUnion.
using GEPConstantIndex =
    llvm::PointerEmbeddedInt<int32_t, kGEPConstantBitWidth>;

/// Class used for building a 'llvm.getelementptr'. A single instance represents
/// a sum type that is either a 'Value' or a constant 'GEPConstantIndex' index.
/// The former represents a dynamic index in a GEP operation, while the later is
/// a constant index as is required for indices into struct types.
class GEPArg : public PointerUnion<Value, GEPConstantIndex> {
  using BaseT = PointerUnion<Value, GEPConstantIndex>;

public:
  /// Constructs a GEPArg with a constant index.
  /*implicit*/ GEPArg(int32_t integer) : BaseT(integer) {}

  /// Constructs a GEPArg with a dynamic index.
  /*implicit*/ GEPArg(Value value) : BaseT(value) {}

  using BaseT::operator=;
};
} // namespace LLVM
} // namespace mlir

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.h.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicOps.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.h.inc"

namespace mlir {
namespace LLVM {

/// Class used for convenient access and iteration over GEP indices.
/// This class is templated to support not only retrieving the dynamic operands
/// of a GEP operation, but also as an adaptor during folding or conversion to
/// LLVM IR.
///
/// GEP indices may either be constant indices or dynamic indices. The
/// 'rawConstantIndices' is specially encoded by GEPOp and contains either the
/// constant index or the information that an index is a dynamic index.
///
/// When an access to such an index is made it is done through the
/// 'DynamicRange' of this class. This way it can be used as getter in GEPOp via
/// 'GEPIndicesAdaptor<ValueRange>' or during folding via
/// 'GEPIndicesAdaptor<ArrayRef<Attribute>>'.
template <typename DynamicRange>
class GEPIndicesAdaptor {
public:
  /// Return type of 'operator[]' and the iterators 'operator*'. It is depended
  /// upon the value type of 'DynamicRange'. If 'DynamicRange' contains
  /// Attributes or subclasses thereof, then value_type is 'Attribute'. In
  /// all other cases it is a pointer union between the value type of
  /// 'DynamicRange' and IntegerAttr.
  using value_type = std::conditional_t<
      std::is_base_of<Attribute,
                      llvm::detail::ValueOfRange<DynamicRange>>::value,
      Attribute,
      PointerUnion<IntegerAttr, llvm::detail::ValueOfRange<DynamicRange>>>;

  /// Constructs a GEPIndicesAdaptor with the raw constant indices of a GEPOp
  /// and the range that is indexed into for retrieving dynamic indices.
  GEPIndicesAdaptor(DenseI32ArrayAttr rawConstantIndices, DynamicRange values)
      : rawConstantIndices(rawConstantIndices), values(std::move(values)) {}

  /// Returns the GEP index at the given position. Note that this operation has
  /// a linear complexity in regards to the accessed position. To iterate over
  /// all indices, use the iterators.
  ///
  /// This operation is invalid if the index is out of bounds.
  value_type operator[](size_t index) const {
    assert(index < size() && "index out of bounds");
    return *std::next(begin(), index);
  }

  /// Returns whether the GEP index at the given position is a dynamic index.
  bool isDynamicIndex(size_t index) const {
    return rawConstantIndices[index] == GEPOp::kDynamicIndex;
  }

  /// Returns the amount of indices of the GEPOp.
  size_t size() const { return rawConstantIndices.size(); }

  /// Returns true if this GEPOp does not have any indices.
  bool empty() const { return rawConstantIndices.empty(); }

  class iterator
      : public llvm::iterator_facade_base<iterator, std::forward_iterator_tag,
                                          value_type, std::ptrdiff_t,
                                          value_type *, value_type> {
  public:
    iterator(const GEPIndicesAdaptor *base,
             ArrayRef<int32_t>::iterator rawConstantIter,
             llvm::detail::IterOfRange<const DynamicRange> valuesIter)
        : base(base), rawConstantIter(rawConstantIter), valuesIter(valuesIter) {
    }

    value_type operator*() const {
      if (*rawConstantIter == GEPOp::kDynamicIndex)
        return *valuesIter;

      return IntegerAttr::get(base->rawConstantIndices.getElementType(),
                              *rawConstantIter);
    }

    iterator &operator++() {
      if (*rawConstantIter == GEPOp::kDynamicIndex)
        valuesIter++;
      rawConstantIter++;
      return *this;
    }

    bool operator==(const iterator &rhs) const {
      return base == rhs.base && rawConstantIter == rhs.rawConstantIter &&
             valuesIter == rhs.valuesIter;
    }

  private:
    const GEPIndicesAdaptor *base;
    ArrayRef<int32_t>::const_iterator rawConstantIter;
    llvm::detail::IterOfRange<const DynamicRange> valuesIter;
  };

  /// Returns the begin iterator, iterating over all GEP indices.
  iterator begin() const {
    return iterator(this, rawConstantIndices.asArrayRef().begin(),
                    values.begin());
  }

  /// Returns the end iterator, iterating over all GEP indices.
  iterator end() const {
    return iterator(this, rawConstantIndices.asArrayRef().end(), values.end());
  }

private:
  DenseI32ArrayAttr rawConstantIndices;
  DynamicRange values;
};

/// Create an LLVM global containing the string "value" at the module containing
/// surrounding the insertion point of builder. Obtain the address of that
/// global and use it to compute the address of the first character in the
/// string (operations inserted at the builder insertion point).
Value createGlobalString(Location loc, OpBuilder &builder, StringRef name,
                         StringRef value, Linkage linkage);

/// LLVM requires some operations to be inside of a Module operation. This
/// function confirms that the Operation has the desired properties.
bool satisfiesLLVMModule(Operation *op);

/// Convert an array of integer attributes to a vector of integers that can be
/// used as indices in LLVM operations.
template <typename IntT = int64_t>
SmallVector<IntT> convertArrayToIndices(ArrayRef<Attribute> attrs) {
  SmallVector<IntT> indices;
  indices.reserve(attrs.size());
  for (Attribute attr : attrs)
    indices.push_back(cast<IntegerAttr>(attr).getInt());
  return indices;
}

/// Convert an `ArrayAttr` of integer attributes to a vector of integers that
/// can be used as indices in LLVM operations.
template <typename IntT = int64_t>
SmallVector<IntT> convertArrayToIndices(ArrayAttr attrs) {
  return convertArrayToIndices<IntT>(attrs.getValue());
}

} // namespace LLVM
} // namespace mlir

namespace llvm {

// Allow llvm::cast style functions.
template <typename To>
struct CastInfo<To, mlir::LLVM::GEPArg>
    : public CastInfo<To, mlir::LLVM::GEPArg::PointerUnion> {};

template <typename To>
struct CastInfo<To, const mlir::LLVM::GEPArg>
    : public CastInfo<To, const mlir::LLVM::GEPArg::PointerUnion> {};

} // namespace llvm

#endif // MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
