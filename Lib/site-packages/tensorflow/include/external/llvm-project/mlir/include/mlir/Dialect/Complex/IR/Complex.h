//===- Complex.h - Complex dialect --------------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_COMPLEX_IR_COMPLEX_H_
#define MLIR_DIALECT_COMPLEX_IR_COMPLEX_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Complex Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Complex/IR/ComplexOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// Complex Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Complex/IR/ComplexAttributes.h.inc"

#endif // MLIR_DIALECT_COMPLEX_IR_COMPLEX_H_
