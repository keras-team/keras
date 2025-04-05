//===- BufferViewFlowOpInterfaceImpl.h - Buffer View Analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_TRANSFORMS_BUFFERVIEWFLOWOPINTERFACEIMPL_H
#define MLIR_DIALECT_MEMREF_TRANSFORMS_BUFFERVIEWFLOWOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace memref {
void registerBufferViewFlowOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace memref
} // namespace mlir

#endif // MLIR_DIALECT_MEMREF_TRANSFORMS_BUFFERVIEWFLOWOPINTERFACEIMPL_H
