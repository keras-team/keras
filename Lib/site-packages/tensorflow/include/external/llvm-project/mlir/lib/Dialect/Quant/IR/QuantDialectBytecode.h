//===- QuantDialectBytecode.h - Quant Bytecode Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header defines hooks into the quantization dialect bytecode
// implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_DIALECT_QUANT_IR_QUANTDIALECTBYTECODE_H
#define LIB_MLIR_DIALECT_QUANT_IR_QUANTDIALECTBYTECODE_H

namespace mlir::quant {
class QuantizationDialect;

namespace detail {
/// Add the interfaces necessary for encoding the quantization dialect
/// components in bytecode.
void addBytecodeInterface(QuantizationDialect *dialect);
} // namespace detail
} // namespace mlir::quant

#endif // LIB_MLIR_DIALECT_QUANT_IR_QUANTDIALECTBYTECODE_H
