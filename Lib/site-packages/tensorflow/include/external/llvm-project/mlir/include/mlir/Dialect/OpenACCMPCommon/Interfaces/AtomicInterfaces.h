//===- DirectiveAtomicInterfaces.h - directive atomic ops interfaces ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interface for atomic operations used
// in OpenACC and OpenMP.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_MP_COMMON_INTERFACES_ATOMICINTERFACES_H_
#define OPENACC_MP_COMMON_INTERFACES_ATOMICINTERFACES_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.h.inc"

#endif // OPENACC_MP_COMMON_INTERFACES_ATOMICINTERFACES_H_
