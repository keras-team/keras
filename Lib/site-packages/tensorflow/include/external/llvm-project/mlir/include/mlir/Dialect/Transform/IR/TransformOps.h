//===- TransformDialect.h - Transform dialect operations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/MatchInterfaces.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

namespace mlir {
namespace transform {

enum class FailurePropagationMode : uint32_t;
class FailurePropagationModeAttr;

/// A builder function that populates the body of a SequenceOp.
using SequenceBodyBuilderFn = ::llvm::function_ref<void(
    ::mlir::OpBuilder &, ::mlir::Location, ::mlir::BlockArgument)>;
using SequenceBodyBuilderArgsFn =
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location,
                              ::mlir::BlockArgument, ::mlir::ValueRange)>;

} // namespace transform
} // namespace mlir

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IR/TransformOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMOPS_H
