//===- Encoding.h - MLIR binary format encoding information -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines enum values describing the structure of MLIR bytecode
// files.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_ENCODING_H
#define MLIR_BYTECODE_ENCODING_H

#include "mlir/IR/Value.h"
#include <cstdint>
#include <type_traits>

namespace mlir {
namespace bytecode {
//===----------------------------------------------------------------------===//
// General constants
//===----------------------------------------------------------------------===//

enum BytecodeVersion {
  /// The minimum supported version of the bytecode.
  kMinSupportedVersion = 0,

  /// Dialects versioning was added in version 1.
  kDialectVersioning = 1,

  /// Support for lazy-loading of isolated region was added in version 2.
  kLazyLoading = 2,

  /// Use-list ordering started to be encoded in version 3.
  kUseListOrdering = 3,

  /// Avoid recording unknown locations on block arguments (compression) started
  /// in version 4.
  kElideUnknownBlockArgLocation = 4,

  /// Support for encoding properties natively in bytecode instead of merged
  /// with the discardable attributes.
  kNativePropertiesEncoding = 5,

  /// ODS emits operand/result segment_size as native properties instead of
  /// an attribute.
  kNativePropertiesODSSegmentSize = 6,

  /// The current bytecode version.
  kVersion = 6,

  /// An arbitrary value used to fill alignment padding.
  kAlignmentByte = 0xCB,
};

//===----------------------------------------------------------------------===//
// Sections
//===----------------------------------------------------------------------===//

namespace Section {
enum ID : uint8_t {
  /// This section contains strings referenced within the bytecode.
  kString = 0,

  /// This section contains the dialects referenced within an IR module.
  kDialect = 1,

  /// This section contains the attributes and types referenced within an IR
  /// module.
  kAttrType = 2,

  /// This section contains the offsets for the attribute and types within the
  /// AttrType section.
  kAttrTypeOffset = 3,

  /// This section contains the list of operations serialized into the bytecode,
  /// and their nested regions/operations.
  kIR = 4,

  /// This section contains the resources of the bytecode.
  kResource = 5,

  /// This section contains the offsets of resources within the Resource
  /// section.
  kResourceOffset = 6,

  /// This section contains the versions of each dialect.
  kDialectVersions = 7,

  /// This section contains the properties for the operations.
  kProperties = 8,

  /// The total number of section types.
  kNumSections = 9,
};
} // namespace Section

//===----------------------------------------------------------------------===//
// IR Section
//===----------------------------------------------------------------------===//

/// This enum represents a mask of all of the potential components of an
/// operation. This mask is used when encoding an operation to indicate which
/// components are present in the bytecode.
namespace OpEncodingMask {
enum : uint8_t {
  // clang-format off
  kHasAttrs         = 0b00000001,
  kHasResults       = 0b00000010,
  kHasOperands      = 0b00000100,
  kHasSuccessors    = 0b00001000,
  kHasInlineRegions = 0b00010000,
  kHasUseListOrders = 0b00100000,
  kHasProperties    = 0b01000000,
  // clang-format on
};
} // namespace OpEncodingMask

/// Get the unique ID of a value use. We encode the unique ID combining an owner
/// number and the argument number such as if ownerID(op1) < ownerID(op2), then
/// useID(op1) < useID(op2). If uses have the same owner, then argNumber(op1) <
/// argNumber(op2) implies useID(op1) < useID(op2).
template <typename OperandT>
static inline uint64_t getUseID(OperandT &val, unsigned ownerID) {
  uint32_t operandNumberID;
  if constexpr (std::is_same_v<OpOperand, OperandT>)
    operandNumberID = val.getOperandNumber();
  else if constexpr (std::is_same_v<BlockArgument, OperandT>)
    operandNumberID = val.getArgNumber();
  else
    llvm_unreachable("unexpected operand type");
  return (static_cast<uint64_t>(ownerID) << 32) | operandNumberID;
}

} // namespace bytecode
} // namespace mlir

#endif
