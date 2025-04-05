//===- StructBuilder.h - Helper for building LLVM structs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a convenience API for emitting IR that inspects or constructs values
// of LLVM dialect structure types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_STRUCTBUILDER_H
#define MLIR_CONVERSION_LLVMCOMMON_STRUCTBUILDER_H

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace mlir {

class OpBuilder;

/// Helper class to produce LLVM dialect operations extracting or inserting
/// values to a struct.
class StructBuilder {
public:
  /// Construct a helper for the given value.
  explicit StructBuilder(Value v);
  /// Builds IR creating an `undef` value of the descriptor type.
  static StructBuilder undef(OpBuilder &builder, Location loc,
                             Type descriptorType);

  /*implicit*/ operator Value() { return value; }

protected:
  // LLVM value
  Value value;
  // Cached struct type.
  Type structType;

protected:
  /// Builds IR to extract a value from the struct at position pos
  Value extractPtr(OpBuilder &builder, Location loc, unsigned pos) const;
  /// Builds IR to set a value in the struct at position pos
  void setPtr(OpBuilder &builder, Location loc, unsigned pos, Value ptr);
};

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMCOMMON_STRUCTBUILDER_H
