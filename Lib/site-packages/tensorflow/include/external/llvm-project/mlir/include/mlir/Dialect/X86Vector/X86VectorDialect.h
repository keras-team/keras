//===- X86VectorDialect.h - MLIR Dialect for X86Vector ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for X86Vector in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_X86VECTOR_X86VECTORDIALECT_H_
#define MLIR_DIALECT_X86VECTOR_X86VECTORDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/X86Vector/X86VectorDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/X86Vector/X86Vector.h.inc"

#endif // MLIR_DIALECT_X86VECTOR_X86VECTORDIALECT_H_
