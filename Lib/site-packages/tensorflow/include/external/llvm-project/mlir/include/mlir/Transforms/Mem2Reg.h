//===-- Mem2Reg.h - Mem2Reg definitions -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_MEM2REG_H
#define MLIR_TRANSFORMS_MEM2REG_H

#include "mlir/Interfaces/MemorySlotInterfaces.h"
#include "llvm/ADT/Statistic.h"

namespace mlir {

/// Statistics collected while applying mem2reg.
struct Mem2RegStatistics {
  /// Total amount of memory slots promoted.
  llvm::Statistic *promotedAmount = nullptr;
  /// Total amount of new block arguments inserted in blocks.
  llvm::Statistic *newBlockArgumentAmount = nullptr;
};

/// Attempts to promote the memory slots of the provided allocators. Iteratively
/// retries the promotion of all slots as promoting one slot might enable
/// subsequent promotions. Succeeds if at least one memory slot was promoted.
LogicalResult
tryToPromoteMemorySlots(ArrayRef<PromotableAllocationOpInterface> allocators,
                        OpBuilder &builder, const DataLayout &dataLayout,
                        DominanceInfo &dominance,
                        Mem2RegStatistics statistics = {});

} // namespace mlir

#endif // MLIR_TRANSFORMS_MEM2REG_H
