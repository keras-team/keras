//===- ComposeSubView.h - Combining composed memref ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Patterns for combining composed subview ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_COMPOSESUBVIEW_H_
#define MLIR_DIALECT_MEMREF_TRANSFORMS_COMPOSESUBVIEW_H_

namespace mlir {
class MLIRContext;
class RewritePatternSet;

namespace memref {

void populateComposeSubViewPatterns(RewritePatternSet &patterns,
                                    MLIRContext *context);

} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_COMPOSESUBVIEW_H_
