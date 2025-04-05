//===- BytecodeOpInterface.h - Bytecode interface for MLIR Op ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the BytecodeOpInterface defined in
// `BytecodeOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_BYTECODE_BYTECODEOPINTERFACE_H
#define MLIR_BYTECODE_BYTECODEOPINTERFACE_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "mlir/Bytecode/BytecodeOpInterface.h.inc"

#endif // MLIR_BYTECODE_BYTECODEOPINTERFACE_H
