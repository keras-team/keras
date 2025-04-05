//===- LLVMTypes.h - MLIR LLVM dialect types --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the LLVM dialect in MLIR. These MLIR types
// correspond to the LLVM IR type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMTYPES_H_
#define MLIR_DIALECT_LLVMIR_LLVMTYPES_H_

#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include <optional>

namespace llvm {
class ElementCount;
class TypeSize;
} // namespace llvm

namespace mlir {

class AsmParser;
class AsmPrinter;

namespace LLVM {
class LLVMDialect;

namespace detail {
struct LLVMFunctionTypeStorage;
struct LLVMPointerTypeStorage;
struct LLVMStructTypeStorage;
struct LLVMTypeAndSizeStorage;
} // namespace detail
} // namespace LLVM
} // namespace mlir

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMTypes.h.inc"

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// Trivial types.
//===----------------------------------------------------------------------===//

// Batch-define trivial types.
#define DEFINE_TRIVIAL_LLVM_TYPE(ClassName, TypeName)                          \
  class ClassName : public Type::TypeBase<ClassName, Type, TypeStorage> {      \
  public:                                                                      \
    using Base::Base;                                                          \
    static constexpr StringLiteral name = TypeName;                            \
  }

DEFINE_TRIVIAL_LLVM_TYPE(LLVMVoidType, "llvm.void");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMPPCFP128Type, "llvm.ppc_fp128");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMTokenType, "llvm.token");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMLabelType, "llvm.label");
DEFINE_TRIVIAL_LLVM_TYPE(LLVMMetadataType, "llvm.metadata");

#undef DEFINE_TRIVIAL_LLVM_TYPE

//===----------------------------------------------------------------------===//
// LLVMStructType.
//===----------------------------------------------------------------------===//

/// LLVM dialect structure type representing a collection of different-typed
/// elements manipulated together. Structured can optionally be packed, meaning
/// that their elements immediately follow each other in memory without
/// accounting for potential alignment.
///
/// Structure types can be identified (named) or literal. Literal structures
/// are uniquely represented by the list of types they contain and packedness.
/// Literal structure types are immutable after construction.
///
/// Identified structures are uniquely represented by their name, a string. They
/// have a mutable component, consisting of the list of types they contain,
/// the packedness and the opacity bits. Identified structs can be created
/// without providing the lists of element types, making them suitable to
/// represent recursive, i.e. self-referring, structures. Identified structs
/// without body are considered opaque. For such structs, one can set the body.
/// Identified structs can be created as intentionally-opaque, implying that the
/// caller does not intend to ever set the body (e.g. forward-declarations of
/// structs from another module) and wants to disallow further modification of
/// the body. For intentionally-opaque structs or non-opaque structs with the
/// body, one is not allowed to set another body (however, one can set exactly
/// the same body).
///
/// Note that the packedness of the struct takes place in uniquing of literal
/// structs, but does not in uniquing of identified structs.
class LLVMStructType
    : public Type::TypeBase<LLVMStructType, Type, detail::LLVMStructTypeStorage,
                            DataLayoutTypeInterface::Trait,
                            DestructurableTypeInterface::Trait,
                            TypeTrait::IsMutable> {
public:
  /// Inherit base constructors.
  using Base::Base;

  static constexpr StringLiteral name = "llvm.struct";

  /// Checks if the given type can be contained in a structure type.
  static bool isValidElementType(Type type);

  /// Gets or creates an identified struct with the given name in the provided
  /// context. Note that unlike llvm::StructType::create, this function will
  /// _NOT_ rename a struct in case a struct with the same name already exists
  /// in the context. Instead, it will just return the existing struct,
  /// similarly to the rest of MLIR type ::get methods.
  static LLVMStructType getIdentified(MLIRContext *context, StringRef name);
  static LLVMStructType
  getIdentifiedChecked(function_ref<InFlightDiagnostic()> emitError,
                       MLIRContext *context, StringRef name);

  /// Gets a new identified struct with the given body. The body _cannot_ be
  /// changed later. If a struct with the given name already exists, renames
  /// the struct by appending a `.` followed by a number to the name. Renaming
  /// happens even if the existing struct has the same body.
  static LLVMStructType getNewIdentified(MLIRContext *context, StringRef name,
                                         ArrayRef<Type> elements,
                                         bool isPacked = false);

  /// Gets or creates a literal struct with the given body in the provided
  /// context.
  static LLVMStructType getLiteral(MLIRContext *context, ArrayRef<Type> types,
                                   bool isPacked = false);
  static LLVMStructType
  getLiteralChecked(function_ref<InFlightDiagnostic()> emitError,
                    MLIRContext *context, ArrayRef<Type> types,
                    bool isPacked = false);

  /// Gets or creates an intentionally-opaque identified struct. Such a struct
  /// cannot have its body set. To create an opaque struct with a mutable body,
  /// use `getIdentified`. Note that unlike llvm::StructType::create, this
  /// function will _NOT_ rename a struct in case a struct with the same name
  /// already exists in the context. Instead, it will just return the existing
  /// struct, similarly to the rest of MLIR type ::get methods.
  static LLVMStructType getOpaque(StringRef name, MLIRContext *context);
  static LLVMStructType
  getOpaqueChecked(function_ref<InFlightDiagnostic()> emitError,
                   MLIRContext *context, StringRef name);

  /// Set the body of an identified struct. Returns failure if the body could
  /// not be set, e.g. if the struct already has a body or if it was marked as
  /// intentionally opaque. This might happen in a multi-threaded context when a
  /// different thread modified the struct after it was created. Most callers
  /// are likely to assert this always succeeds, but it is possible to implement
  /// a local renaming scheme based on the result of this call.
  LogicalResult setBody(ArrayRef<Type> types, bool isPacked);

  /// Checks if a struct is packed.
  bool isPacked() const;

  /// Checks if a struct is identified.
  bool isIdentified() const;

  /// Checks if a struct is opaque.
  bool isOpaque();

  /// Checks if a struct is initialized.
  bool isInitialized();

  /// Returns the name of an identified struct.
  StringRef getName();

  /// Returns the list of element types contained in a non-opaque struct.
  ArrayRef<Type> getBody() const;

  /// Verifies that the type about to be constructed is well-formed.
  static LogicalResult
  verifyInvariants(function_ref<InFlightDiagnostic()> emitError, StringRef,
                   bool);
  static LogicalResult
  verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Type> types, bool);
  using Base::verifyInvariants;

  /// Hooks for DataLayoutTypeInterface. Should not be called directly. Obtain a
  /// DataLayout instance and query it instead.
  llvm::TypeSize getTypeSizeInBits(const DataLayout &dataLayout,
                                   DataLayoutEntryListRef params) const;

  uint64_t getABIAlignment(const DataLayout &dataLayout,
                           DataLayoutEntryListRef params) const;

  uint64_t getPreferredAlignment(const DataLayout &dataLayout,
                                 DataLayoutEntryListRef params) const;

  bool areCompatible(DataLayoutEntryListRef oldLayout,
                     DataLayoutEntryListRef newLayout) const;

  LogicalResult verifyEntries(DataLayoutEntryListRef entries,
                              Location loc) const;

  /// Destructs the struct into its indexed field types.
  std::optional<DenseMap<Attribute, Type>> getSubelementIndexMap();

  /// Returns which type is stored at a given integer index within the struct.
  Type getTypeAtIndex(Attribute index);
};

//===----------------------------------------------------------------------===//
// Printing and parsing.
//===----------------------------------------------------------------------===//

namespace detail {
/// Parses an LLVM dialect type.
Type parseType(DialectAsmParser &parser);

/// Prints an LLVM Dialect type.
void printType(Type type, AsmPrinter &printer);
} // namespace detail

/// Parse any MLIR type or a concise syntax for LLVM types.
ParseResult parsePrettyLLVMType(AsmParser &p, Type &type);
/// Print any MLIR type or a concise syntax for LLVM types.
void printPrettyLLVMType(AsmPrinter &p, Type type);

//===----------------------------------------------------------------------===//
// Utility functions.
//===----------------------------------------------------------------------===//

/// Returns `true` if the given type is compatible with the LLVM dialect. This
/// is an alias to `LLVMDialect::isCompatibleType`.
bool isCompatibleType(Type type);

/// Returns `true` if the given outer type is compatible with the LLVM dialect
/// without checking its potential nested types such as struct elements.
bool isCompatibleOuterType(Type type);

/// Returns `true` if the given type is a floating-point type compatible with
/// the LLVM dialect.
bool isCompatibleFloatingPointType(Type type);

/// Returns `true` if the given type is a vector type compatible with the LLVM
/// dialect. Compatible types include 1D built-in vector types of built-in
/// integers and floating-point values, LLVM dialect fixed vector types of LLVM
/// dialect pointers and LLVM dialect scalable vector types.
bool isCompatibleVectorType(Type type);

/// Returns the element type of any vector type compatible with the LLVM
/// dialect.
Type getVectorElementType(Type type);

/// Returns the element count of any LLVM-compatible vector type.
llvm::ElementCount getVectorNumElements(Type type);

/// Returns whether a vector type is scalable or not.
bool isScalableVectorType(Type vectorType);

/// Creates an LLVM dialect-compatible vector type with the given element type
/// and length.
Type getVectorType(Type elementType, unsigned numElements,
                   bool isScalable = false);

/// Creates an LLVM dialect-compatible vector type with the given element type
/// and length.
Type getVectorType(Type elementType, const llvm::ElementCount &numElements);

/// Creates an LLVM dialect-compatible type with the given element type and
/// length.
Type getFixedVectorType(Type elementType, unsigned numElements);

/// Creates an LLVM dialect-compatible type with the given element type and
/// length.
Type getScalableVectorType(Type elementType, unsigned numElements);

/// Returns the size of the given primitive LLVM dialect-compatible type
/// (including vectors) in bits, for example, the size of i16 is 16 and
/// the size of vector<4xi16> is 64. Returns 0 for non-primitive
/// (aggregates such as struct) or types that don't have a size (such as void).
llvm::TypeSize getPrimitiveTypeSizeInBits(Type type);

/// The positions of different values in the data layout entry for pointers.
enum class PtrDLEntryPos { Size = 0, Abi = 1, Preferred = 2, Index = 3 };

/// Returns the value that corresponds to named position `pos` from the
/// data layout entry `attr` assuming it's a dense integer elements attribute.
/// Returns `std::nullopt` if `pos` is not present in the entry.
/// Currently only `PtrDLEntryPos::Index` is optional, and all other positions
/// may be assumed to be present.
std::optional<uint64_t> extractPointerSpecValue(Attribute attr,
                                                PtrDLEntryPos pos);

} // namespace LLVM
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMTYPES_H_
