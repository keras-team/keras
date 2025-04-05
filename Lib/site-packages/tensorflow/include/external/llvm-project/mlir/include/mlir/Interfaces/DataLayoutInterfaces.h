//===- DataLayoutInterfaces.h - Data Layout Interface Decls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces for the data layout specification, operations to which
// they can be attached, types subject to data layout and dialects containing
// data layout entries.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_DATALAYOUTINTERFACES_H
#define MLIR_INTERFACES_DATALAYOUTINTERFACES_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/TypeSize.h"

namespace mlir {
class DataLayout;
class DataLayoutEntryInterface;
class DLTIQueryInterface;
class TargetDeviceSpecInterface;
class TargetSystemSpecInterface;
using DataLayoutEntryKey = llvm::PointerUnion<Type, StringAttr>;
// Using explicit SmallVector size because we cannot infer the size from the
// forward declaration, and we need the typedef in the actual declaration.
using DataLayoutEntryList = llvm::SmallVector<DataLayoutEntryInterface, 4>;
using DataLayoutEntryListRef = llvm::ArrayRef<DataLayoutEntryInterface>;
using TargetDeviceSpecListRef = llvm::ArrayRef<TargetDeviceSpecInterface>;
using DeviceIDTargetDeviceSpecPair =
    std::pair<StringAttr, TargetDeviceSpecInterface>;
using DeviceIDTargetDeviceSpecPairListRef =
    llvm::ArrayRef<DeviceIDTargetDeviceSpecPair>;
class DataLayoutOpInterface;
class DataLayoutSpecInterface;
class ModuleOp;

namespace detail {
/// Default handler for the type size request. Computes results for built-in
/// types and dispatches to the DataLayoutTypeInterface for other types.
llvm::TypeSize getDefaultTypeSize(Type type, const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params);

/// Default handler for the type size in bits request. Computes results for
/// built-in types and dispatches to the DataLayoutTypeInterface for other
/// types.
llvm::TypeSize getDefaultTypeSizeInBits(Type type, const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params);

/// Default handler for the required alignment request. Computes results for
/// built-in types and dispatches to the DataLayoutTypeInterface for other
/// types.
uint64_t getDefaultABIAlignment(Type type, const DataLayout &dataLayout,
                                ArrayRef<DataLayoutEntryInterface> params);

/// Default handler for the preferred alignment request. Computes results for
/// built-in types and dispatches to the DataLayoutTypeInterface for other
/// types.
uint64_t
getDefaultPreferredAlignment(Type type, const DataLayout &dataLayout,
                             ArrayRef<DataLayoutEntryInterface> params);

/// Default handler for the index bitwidth request. Computes the result for
/// the built-in index type and dispatches to the DataLayoutTypeInterface for
/// other types.
std::optional<uint64_t>
getDefaultIndexBitwidth(Type type, const DataLayout &dataLayout,
                        ArrayRef<DataLayoutEntryInterface> params);

/// Default handler for endianness request. Dispatches to the
/// DataLayoutInterface if specified, otherwise returns the default.
Attribute getDefaultEndianness(DataLayoutEntryInterface entry);

/// Default handler for alloca memory space request. Dispatches to the
/// DataLayoutInterface if specified, otherwise returns the default.
Attribute getDefaultAllocaMemorySpace(DataLayoutEntryInterface entry);

/// Default handler for program memory space request. Dispatches to the
/// DataLayoutInterface if specified, otherwise returns the default.
Attribute getDefaultProgramMemorySpace(DataLayoutEntryInterface entry);

/// Default handler for global memory space request. Dispatches to the
/// DataLayoutInterface if specified, otherwise returns the default.
Attribute getDefaultGlobalMemorySpace(DataLayoutEntryInterface entry);

/// Default handler for the stack alignment request. Dispatches to the
/// DataLayoutInterface if specified, otherwise returns the default.
uint64_t getDefaultStackAlignment(DataLayoutEntryInterface entry);

/// Returns the value of the property from the specified DataLayoutEntry. If the
/// property is missing from the entry, returns std::nullopt.
std::optional<Attribute> getDevicePropertyValue(DataLayoutEntryInterface entry);

/// Given a list of data layout entries, returns a new list containing the
/// entries with keys having the given type ID, i.e. belonging to the same type
/// class.
DataLayoutEntryList filterEntriesForType(DataLayoutEntryListRef entries,
                                         TypeID typeID);

/// Given a list of data layout entries, returns the entry that has the given
/// identifier as key, if such an entry exists in the list.
DataLayoutEntryInterface
filterEntryForIdentifier(DataLayoutEntryListRef entries, StringAttr id);

/// Given a list of target device entries, returns the entry that has the given
/// identifier as key, if such an entry exists in the list.
TargetDeviceSpecInterface
filterEntryForIdentifier(TargetDeviceSpecListRef entries, StringAttr id);

/// Verifies that the operation implementing the data layout interface, or a
/// module operation, is valid. This calls the verifier of the spec attribute
/// and checks if the layout is compatible with specs attached to the enclosing
/// operations.
LogicalResult verifyDataLayoutOp(Operation *op);

/// Verifies that a data layout spec is valid. This dispatches to individual
/// entry verifiers, and then to the verifiers implemented by the relevant type
/// and dialect interfaces for type and identifier keys respectively.
LogicalResult verifyDataLayoutSpec(DataLayoutSpecInterface spec, Location loc);

/// Verifies that a target system desc spec is valid. This dispatches to
/// individual entry verifiers, and then to the verifiers implemented by the
/// relevant dialect interfaces for identifier keys.
LogicalResult verifyTargetSystemSpec(TargetSystemSpecInterface spec,
                                     Location loc);

/// Divides the known min value of the numerator by the denominator and rounds
/// the result up to the next integer. Preserves the scalable flag.
llvm::TypeSize divideCeil(llvm::TypeSize numerator, uint64_t denominator);
} // namespace detail
} // namespace mlir

#include "mlir/Interfaces/DataLayoutAttrInterface.h.inc"
#include "mlir/Interfaces/DataLayoutOpInterface.h.inc"
#include "mlir/Interfaces/DataLayoutTypeInterface.h.inc"

namespace mlir {

//===----------------------------------------------------------------------===//
// DataLayoutDialectInterface
//===----------------------------------------------------------------------===//

/// An interface to be implemented by dialects that can have identifiers in the
/// data layout specification entries. Provides hooks for verifying the entry
/// validity and combining two entries.
class DataLayoutDialectInterface
    : public DialectInterface::Base<DataLayoutDialectInterface> {
public:
  DataLayoutDialectInterface(Dialect *dialect) : Base(dialect) {}

  /// Checks whether the given data layout entry is valid and reports any errors
  /// at the provided location. Derived classes should override this.
  virtual LogicalResult verifyEntry(DataLayoutEntryInterface entry,
                                    Location loc) const {
    return success();
  }

  /// Checks whether the given data layout entry is valid and reports any errors
  /// at the provided location. Derived classes should override this.
  virtual LogicalResult verifyEntry(TargetDeviceSpecInterface entry,
                                    Location loc) const {
    return success();
  }

  /// Default implementation of entry combination that combines identical
  /// entries and returns null otherwise.
  static DataLayoutEntryInterface
  defaultCombine(DataLayoutEntryInterface outer,
                 DataLayoutEntryInterface inner) {
    if (!outer || outer == inner)
      return inner;
    return {};
  }

  /// Combines two entries with identifiers that belong to this dialect. Returns
  /// the combined entry or null if the entries are not compatible. Derived
  /// classes likely need to reimplement this.
  virtual DataLayoutEntryInterface
  combine(DataLayoutEntryInterface outer,
          DataLayoutEntryInterface inner) const {
    return defaultCombine(outer, inner);
  }
};

//===----------------------------------------------------------------------===//
// DataLayout
//===----------------------------------------------------------------------===//

/// The main mechanism for performing data layout queries. Instances of this
/// class can be created for an operation implementing DataLayoutOpInterface.
/// Upon construction, a layout spec combining that of the given operation with
/// all its ancestors will be computed and used to handle further requests. For
/// efficiency, results to all requests will be cached in this object.
/// Therefore, if the data layout spec for the scoping operation, or any of the
/// enclosing operations, changes, the cache is no longer valid. The user is
/// responsible creating a new DataLayout object after any spec change. In debug
/// mode, the cache validity is being checked in every request.
class DataLayout {
public:
  explicit DataLayout();
  explicit DataLayout(DataLayoutOpInterface op);
  explicit DataLayout(ModuleOp op);

  /// Returns the layout of the closest parent operation carrying layout info.
  static DataLayout closest(Operation *op);

  /// Returns the size of the given type in the current scope.
  llvm::TypeSize getTypeSize(Type t) const;

  /// Returns the size in bits of the given type in the current scope.
  llvm::TypeSize getTypeSizeInBits(Type t) const;

  /// Returns the required alignment of the given type in the current scope.
  uint64_t getTypeABIAlignment(Type t) const;

  /// Returns the preferred of the given type in the current scope.
  uint64_t getTypePreferredAlignment(Type t) const;

  /// Returns the bitwidth that should be used when performing index
  /// computations for the given pointer-like type in the current scope. If the
  /// type is not a pointer-like type, it returns std::nullopt.
  std::optional<uint64_t> getTypeIndexBitwidth(Type t) const;

  /// Returns the specified endianness.
  Attribute getEndianness() const;

  /// Returns the memory space used for AllocaOps.
  Attribute getAllocaMemorySpace() const;

  /// Returns the memory space used for program memory operations.
  Attribute getProgramMemorySpace() const;

  /// Returns the memory space used for global operations.
  Attribute getGlobalMemorySpace() const;

  /// Returns the natural alignment of the stack in bits. Alignment promotion of
  /// stack variables should be limited to the natural stack alignment to
  /// prevent dynamic stack alignment. Returns zero if the stack alignment is
  /// unspecified.
  uint64_t getStackAlignment() const;

  /// Returns the value of the specified property if the property is defined for
  /// the given device ID, otherwise returns std::nullopt.
  std::optional<Attribute>
  getDevicePropertyValue(TargetSystemSpecInterface::DeviceID,
                         StringAttr propertyName) const;

private:
  /// Combined layout spec at the given scope.
  const DataLayoutSpecInterface originalLayout;

  /// Combined target system desc spec at the given scope.
  const TargetSystemSpecInterface originalTargetSystemDesc;

#if LLVM_ENABLE_ABI_BREAKING_CHECKS
  /// List of enclosing layout specs.
  SmallVector<DataLayoutSpecInterface, 2> layoutStack;
#endif

  /// Asserts that the cache is still valid. Expensive in debug mode. No-op in
  /// release mode.
  void checkValid() const;

  /// Operation defining the scope of requests.
  Operation *scope;

  /// Caches for individual requests.
  mutable DenseMap<Type, llvm::TypeSize> sizes;
  mutable DenseMap<Type, llvm::TypeSize> bitsizes;
  mutable DenseMap<Type, uint64_t> abiAlignments;
  mutable DenseMap<Type, uint64_t> preferredAlignments;
  mutable DenseMap<Type, std::optional<uint64_t>> indexBitwidths;

  /// Cache for the endianness.
  mutable std::optional<Attribute> endianness;
  /// Cache for alloca, global, and program memory spaces.
  mutable std::optional<Attribute> allocaMemorySpace;
  mutable std::optional<Attribute> programMemorySpace;
  mutable std::optional<Attribute> globalMemorySpace;

  /// Cache for stack alignment.
  mutable std::optional<uint64_t> stackAlignment;
};

} // namespace mlir

#endif // MLIR_INTERFACES_DATALAYOUTINTERFACES_H
