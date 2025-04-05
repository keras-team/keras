//===- MLProgramAttributes.h - Attribute Classes ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MLPROGRAM_IR_MLPROGRAMATTRIBUTES_H_
#define MLIR_DIALECT_MLPROGRAM_IR_MLPROGRAMATTRIBUTES_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h.inc"

#endif // MLIR_DIALECT_MLPROGRAM_IR_MLPROGRAMATTRIBUTES_H_
