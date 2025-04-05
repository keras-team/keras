//===- BufferViewFlowOpInterface.h - Buffer View Flow Analysis --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERVIEWFLOWOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERVIEWFLOWOPINTERFACE_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class ValueRange;

namespace bufferization {

using RegisterDependenciesFn = std::function<void(ValueRange, ValueRange)>;

} // namespace bufferization
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/BufferViewFlowOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERVIEWFLOWOPINTERFACE_H_
