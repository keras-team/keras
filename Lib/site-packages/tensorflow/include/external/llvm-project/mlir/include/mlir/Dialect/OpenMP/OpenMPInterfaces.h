//===- OpenMPInterfaces.h - MLIR Interfaces for OpenMP ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares OpenMP Interface implementations for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPINTERFACES_H_
#define MLIR_DIALECT_OPENMP_OPENMPINTERFACES_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_FWD_DEFINES
#include "mlir/Dialect/OpenMP/OpenMPOps.h.inc"

#include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.h.inc"

namespace mlir::omp {
// You can override defaults here or implement more complex implementations of
// functions. Or define a completely separate external model implementation,
// to override the existing implementation.
struct OffloadModuleDefaultModel
    : public OffloadModuleInterface::ExternalModel<OffloadModuleDefaultModel,
                                                   mlir::ModuleOp> {};

template <typename T>
struct DeclareTargetDefaultModel
    : public DeclareTargetInterface::ExternalModel<DeclareTargetDefaultModel<T>,
                                                   T> {};

} // namespace mlir::omp

#endif // MLIR_DIALECT_OPENMP_OPENMPINTERFACES_H_
