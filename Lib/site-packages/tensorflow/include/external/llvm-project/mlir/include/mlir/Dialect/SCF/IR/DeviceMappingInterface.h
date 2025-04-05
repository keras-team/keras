//===- DeviceMappingInterface.h - -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the device mapping interface defined in
// `DeviceMappingInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DEVICEMAPPINGINTERFACE_H
#define MLIR_DEVICEMAPPINGINTERFACE_H

#include "mlir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "mlir/Dialect/SCF/IR/DeviceMappingAttrInterface.h.inc"

#endif // MLIR_DEVICEMAPPINGINTERFACE_H
