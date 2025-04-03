//===- OpenMPDialect.h - MLIR Dialect for OpenMP ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the OpenMP dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_
#define MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h"
#include "mlir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsTypes.h.inc"

#include "mlir/Dialect/OpenMP/OpenMPOpsDialect.h.inc"

#include "mlir/Dialect/OpenMP/OpenMPClauseOperands.h"

#include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.h.inc"

#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOps.h.inc"

#endif // MLIR_DIALECT_OPENMP_OPENMPDIALECT_H_
