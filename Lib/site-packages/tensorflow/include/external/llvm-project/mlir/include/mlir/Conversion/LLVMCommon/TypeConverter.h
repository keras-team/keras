//===- TypeConverter.h - Convert builtin to LLVM dialect types --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a type converter configuration for converting most builtin types to
// LLVM dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_TYPECONVERTER_H
#define MLIR_CONVERSION_LLVMCOMMON_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class DataLayoutAnalysis;
class FunctionOpInterface;
class LowerToLLVMOptions;

namespace LLVM {
class LLVMDialect;
class LLVMPointerType;
class LLVMFunctionType;
class LLVMStructType;
} // namespace LLVM

/// Conversion from types to the LLVM IR dialect.
class LLVMTypeConverter : public TypeConverter {
  /// Give structFuncArgTypeConverter access to memref-specific functions.
  friend LogicalResult
  structFuncArgTypeConverter(const LLVMTypeConverter &converter, Type type,
                             SmallVectorImpl<Type> &result);

public:
  using TypeConverter::convertType;

  /// Create an LLVMTypeConverter using the default LowerToLLVMOptions.
  /// Optionally takes a data layout analysis to use in conversions.
  LLVMTypeConverter(MLIRContext *ctx,
                    const DataLayoutAnalysis *analysis = nullptr);

  /// Create an LLVMTypeConverter using custom LowerToLLVMOptions. Optionally
  /// takes a data layout analysis to use in conversions.
  LLVMTypeConverter(MLIRContext *ctx, const LowerToLLVMOptions &options,
                    const DataLayoutAnalysis *analysis = nullptr);

  /// Convert a function type. The arguments and results are converted one by
  /// one and results are packed into a wrapped LLVM IR structure type. `result`
  /// is populated with argument mapping.
  Type convertFunctionSignature(FunctionType funcTy, bool isVariadic,
                                bool useBarePtrCallConv,
                                SignatureConversion &result) const;

  /// Convert a function type. The arguments and results are converted one by
  /// one and results are packed into a wrapped LLVM IR structure type. `result`
  /// is populated with argument mapping. Converted types of `llvm.byval` and
  /// `llvm.byref` function arguments which are not LLVM pointers are overridden
  /// with LLVM pointers. Overridden arguments are returned in
  /// `byValRefNonPtrAttrs`.
  Type convertFunctionSignature(FunctionOpInterface funcOp, bool isVariadic,
                                bool useBarePtrCallConv,
                                LLVMTypeConverter::SignatureConversion &result,
                                SmallVectorImpl<std::optional<NamedAttribute>>
                                    &byValRefNonPtrAttrs) const;

  /// Convert a non-empty list of types to be returned from a function into an
  /// LLVM-compatible type. In particular, if more than one value is returned,
  /// create an LLVM dialect structure type with elements that correspond to
  /// each of the types converted with `convertCallingConventionType`.
  Type packFunctionResults(TypeRange types,
                           bool useBarePointerCallConv = false) const;

  /// Convert a non-empty list of types of values produced by an operation into
  /// an LLVM-compatible type. In particular, if more than one value is
  /// produced, create a literal structure with elements that correspond to each
  /// of the LLVM-compatible types converted with `convertType`.
  Type packOperationResults(TypeRange types) const;

  /// Convert a type in the context of the default or bare pointer calling
  /// convention. Calling convention sensitive types, such as MemRefType and
  /// UnrankedMemRefType, are converted following the specific rules for the
  /// calling convention. Calling convention independent types are converted
  /// following the default LLVM type conversions.
  Type convertCallingConventionType(Type type,
                                    bool useBarePointerCallConv = false) const;

  /// Promote the bare pointers in 'values' that resulted from memrefs to
  /// descriptors. 'stdTypes' holds the types of 'values' before the conversion
  /// to the LLVM-IR dialect (i.e., MemRefType, or any other builtin type).
  void promoteBarePtrsToDescriptors(ConversionPatternRewriter &rewriter,
                                    Location loc, ArrayRef<Type> stdTypes,
                                    SmallVectorImpl<Value> &values) const;

  /// Returns the MLIR context.
  MLIRContext &getContext() const;

  /// Returns the LLVM dialect.
  LLVM::LLVMDialect *getDialect() const { return llvmDialect; }

  const LowerToLLVMOptions &getOptions() const { return options; }

  /// Promote the LLVM representation of all operands including promoting MemRef
  /// descriptors to stack and use pointers to struct to avoid the complexity
  /// of the platform-specific C/C++ ABI lowering related to struct argument
  /// passing.
  SmallVector<Value, 4> promoteOperands(Location loc, ValueRange opOperands,
                                        ValueRange operands, OpBuilder &builder,
                                        bool useBarePtrCallConv = false) const;

  /// Promote the LLVM struct representation of one MemRef descriptor to stack
  /// and use pointer to struct to avoid the complexity of the platform-specific
  /// C/C++ ABI lowering related to struct argument passing.
  Value promoteOneMemRefDescriptor(Location loc, Value operand,
                                   OpBuilder &builder) const;

  /// Converts the function type to a C-compatible format, in particular using
  /// pointers to memref descriptors for arguments. Also converts the return
  /// type to a pointer argument if it is a struct. Returns true if this
  /// was the case.
  std::pair<LLVM::LLVMFunctionType, LLVM::LLVMStructType>
  convertFunctionTypeCWrapper(FunctionType type) const;

  /// Returns the data layout to use during and after conversion.
  const llvm::DataLayout &getDataLayout() const { return options.dataLayout; }

  /// Returns the data layout analysis to query during conversion.
  const DataLayoutAnalysis *getDataLayoutAnalysis() const {
    return dataLayoutAnalysis;
  }

  /// Gets the LLVM representation of the index type. The returned type is an
  /// integer type with the size configured for this type converter.
  Type getIndexType() const;

  /// Gets the bitwidth of the index type when converted to LLVM.
  unsigned getIndexTypeBitwidth() const { return options.getIndexBitwidth(); }

  /// Gets the pointer bitwidth.
  unsigned getPointerBitwidth(unsigned addressSpace = 0) const;

  /// Returns the size of the memref descriptor object in bytes.
  unsigned getMemRefDescriptorSize(MemRefType type,
                                   const DataLayout &layout) const;

  /// Returns the size of the unranked memref descriptor object in bytes.
  unsigned getUnrankedMemRefDescriptorSize(UnrankedMemRefType type,
                                           const DataLayout &layout) const;

  /// Return the LLVM address space corresponding to the memory space of the
  /// memref type `type` or failure if the memory space cannot be converted to
  /// an integer.
  FailureOr<unsigned> getMemRefAddressSpace(BaseMemRefType type) const;

  /// Check if a memref type can be converted to a bare pointer.
  static bool canConvertToBarePtr(BaseMemRefType type);

protected:
  /// Pointer to the LLVM dialect.
  LLVM::LLVMDialect *llvmDialect;

  // Recursive structure detection.
  // We store one entry per thread here, and rely on locking.
  DenseMap<uint64_t, std::unique_ptr<SmallVector<Type>>> conversionCallStack;
  llvm::sys::SmartRWMutex<true> callStackMutex;
  SmallVector<Type> &getCurrentThreadRecursiveStack();

private:
  /// Convert a function type. The arguments and results are converted one by
  /// one. Additionally, if the function returns more than one value, pack the
  /// results into an LLVM IR structure type so that the converted function type
  /// returns at most one result.
  Type convertFunctionType(FunctionType type) const;

  /// Common implementation for `convertFunctionSignature` methods. Convert a
  /// function type. The arguments and results are converted one by one and
  /// results are packed into a wrapped LLVM IR structure type. `result` is
  /// populated with argument mapping. If `byValRefNonPtrAttrs` is provided,
  /// converted types of `llvm.byval` and `llvm.byref` function arguments which
  /// are not LLVM pointers are overridden with LLVM pointers. `llvm.byval` and
  /// `llvm.byref` arguments that were already converted to LLVM pointer types
  /// are removed from 'byValRefNonPtrAttrs`.
  Type convertFunctionSignatureImpl(
      FunctionType funcTy, bool isVariadic, bool useBarePtrCallConv,
      LLVMTypeConverter::SignatureConversion &result,
      SmallVectorImpl<std::optional<NamedAttribute>> *byValRefNonPtrAttrs)
      const;

  /// Convert the index type.  Uses llvmModule data layout to create an integer
  /// of the pointer bitwidth.
  Type convertIndexType(IndexType type) const;

  /// Convert an integer type `i*` to `!llvm<"i*">`.
  Type convertIntegerType(IntegerType type) const;

  /// Convert a floating point type: `f16` to `f16`, `f32` to
  /// `f32` and `f64` to `f64`.  `bf16` is not supported
  /// by LLVM. 8-bit float types are converted to 8-bit integers as this is how
  /// all LLVM backends that support them currently represent them.
  Type convertFloatType(FloatType type) const;

  /// Convert complex number type: `complex<f16>` to `!llvm<"{ half, half }">`,
  /// `complex<f32>` to `!llvm<"{ float, float }">`, and `complex<f64>` to
  /// `!llvm<"{ double, double }">`. `complex<bf16>` is not supported.
  Type convertComplexType(ComplexType type) const;

  /// Convert a memref type into an LLVM type that captures the relevant data.
  Type convertMemRefType(MemRefType type) const;

  /// Convert a memref type into a list of LLVM IR types that will form the
  /// memref descriptor. If `unpackAggregates` is true the `sizes` and `strides`
  /// arrays in the descriptors are unpacked to individual index-typed elements,
  /// else they are kept as rank-sized arrays of index type. In particular,
  /// the list will contain:
  /// - two pointers to the memref element type, followed by
  /// - an index-typed offset, followed by
  /// - (if unpackAggregates = true)
  ///    - one index-typed size per dimension of the memref, followed by
  ///    - one index-typed stride per dimension of the memref.
  /// - (if unpackArrregates = false)
  ///   - one rank-sized array of index-type for the size of each dimension
  ///   - one rank-sized array of index-type for the stride of each dimension
  ///
  /// For example, memref<?x?xf32> is converted to the following list:
  /// - `!llvm<"float*">` (allocated pointer),
  /// - `!llvm<"float*">` (aligned pointer),
  /// - `i64` (offset),
  /// - `i64`, `i64` (sizes),
  /// - `i64`, `i64` (strides).
  /// These types can be recomposed to a memref descriptor struct.
  SmallVector<Type, 5> getMemRefDescriptorFields(MemRefType type,
                                                 bool unpackAggregates) const;

  /// Convert an unranked memref type into a list of non-aggregate LLVM IR types
  /// that will form the unranked memref descriptor. In particular, this list
  /// contains:
  /// - an integer rank, followed by
  /// - a pointer to the memref descriptor struct.
  /// For example, memref<*xf32> is converted to the following list:
  /// i64 (rank)
  /// !llvm<"i8*"> (type-erased pointer).
  /// These types can be recomposed to a unranked memref descriptor struct.
  SmallVector<Type, 2> getUnrankedMemRefDescriptorFields() const;

  /// Convert an unranked memref type to an LLVM type that captures the
  /// runtime rank and a pointer to the static ranked memref desc
  Type convertUnrankedMemRefType(UnrankedMemRefType type) const;

  /// Convert a memref type to a bare pointer to the memref element type.
  Type convertMemRefToBarePtr(BaseMemRefType type) const;

  /// Convert a 1D vector type into an LLVM vector type.
  FailureOr<Type> convertVectorType(VectorType type) const;

  /// Options for customizing the llvm lowering.
  LowerToLLVMOptions options;

  /// Data layout analysis mapping scopes to layouts active in them.
  const DataLayoutAnalysis *dataLayoutAnalysis;
};

/// Callback to convert function argument types. It converts a MemRef function
/// argument to a list of non-aggregate types containing descriptor
/// information, and an UnrankedmemRef function argument to a list containing
/// the rank and a pointer to a descriptor struct.
LogicalResult structFuncArgTypeConverter(const LLVMTypeConverter &converter,
                                         Type type,
                                         SmallVectorImpl<Type> &result);

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult barePtrFuncArgTypeConverter(const LLVMTypeConverter &converter,
                                          Type type,
                                          SmallVectorImpl<Type> &result);

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMCOMMON_TYPECONVERTER_H
