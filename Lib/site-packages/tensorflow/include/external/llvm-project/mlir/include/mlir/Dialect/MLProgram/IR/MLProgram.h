//===- MLProgram.h - MLProgram dialect ----------------------------*- C++-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MLPROGRAM_IR_MLPROGRAM_H_
#define MLIR_DIALECT_MLPROGRAM_IR_MLPROGRAM_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgramTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// MLProgramDialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/IR/MLProgramOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// MLProgram Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MLProgram/IR/MLProgramOps.h.inc"

#endif // MLIR_DIALECT_MLPROGRAM_IR_MLPROGRAM_H_
