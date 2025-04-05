//===- Passes.h - Transform dialect pass entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;

namespace transform {
#define GEN_PASS_DECL
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES_H
