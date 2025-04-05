//===- UBOps.h - UB Dialect Operations ------------------------*--- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UB_IR_OPS_H
#define MLIR_DIALECT_UB_IR_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/UB/IR/UBOpsInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/UB/IR/UBOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/UB/IR/UBOps.h.inc"

#include "mlir/Dialect/UB/IR/UBOpsDialect.h.inc"

#endif // MLIR_DIALECT_UB_IR_OPS_H
