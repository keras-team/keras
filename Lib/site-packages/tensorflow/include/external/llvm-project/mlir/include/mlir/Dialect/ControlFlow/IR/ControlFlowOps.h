//===- ControlFlowOps.h - ControlFlow Operations ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations of the ControlFlow dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CONTROLFLOW_IR_CONTROLFLOWOPS_H
#define MLIR_DIALECT_CONTROLFLOW_IR_CONTROLFLOWOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
class PatternRewriter;
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h.inc"

#endif // MLIR_DIALECT_CONTROLFLOW_IR_CONTROLFLOWOPS_H
