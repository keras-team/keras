//===- ArmSVEDialect.h - MLIR Dialect for Arm SVE ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for ArmSVE in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_ARMSVEDIALECT_H
#define MLIR_DIALECT_ARMSVE_ARMSVEDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/ArmSVE/IR/ArmSVE.h.inc"

#endif // MLIR_DIALECT_ARMSVE_ARMSVEDIALECT_H
