//===- QuantOps.h - Quantization Ops and Types ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_QUANT_QUANTOPS_H_
#define MLIR_DIALECT_QUANT_QUANTOPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/Dialect/Quant/QuantOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/QuantOps.h.inc"

#endif // MLIR_DIALECT_QUANT_QUANTOPS_H_
