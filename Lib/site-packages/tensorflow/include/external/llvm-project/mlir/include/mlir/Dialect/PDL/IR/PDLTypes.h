//===- PDLTypes.h - Pattern Descriptor Language Types -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types for the Pattern Descriptor Language dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PDL_IR_PDLTYPES_H_
#define MLIR_DIALECT_PDL_IR_PDLTYPES_H_

#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// PDL Dialect Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl {
/// This class represents the base class of all PDL types.
class PDLType : public Type {
public:
  using Type::Type;

  static bool classof(Type type);
};

/// If the given type is a range, return its element type, otherwise return
/// the type itself.
Type getRangeElementTypeOrSelf(Type type);

} // namespace pdl
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/PDL/IR/PDLOpsTypes.h.inc"

#endif // MLIR_DIALECT_PDL_IR_PDLTYPES_H_
