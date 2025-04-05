//===- CFGLoopInfo.h - LoopInfo analysis for region bodies ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the CFGLoopInfo analysis for MLIR. The CFGLoopInfo is used
// to identify natural loops and determine the loop depth of various nodes of a
// CFG.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_LOOPINFO_H
#define MLIR_ANALYSIS_LOOPINFO_H

#include "mlir/IR/Dominance.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "llvm/Support/GenericLoopInfo.h"

namespace mlir {
class CFGLoop;
class CFGLoopInfo;
} // namespace mlir

namespace llvm {
// Implementation in LLVM's LoopInfoImpl.h
extern template class LoopBase<mlir::Block, mlir::CFGLoop>;
extern template class LoopInfoBase<mlir::Block, mlir::CFGLoop>;
} // namespace llvm

namespace mlir {

/// Representation of a single loop formed by blocks. The inherited LoopBase
/// class provides accessors to the loop analysis.
class CFGLoop : public llvm::LoopBase<mlir::Block, mlir::CFGLoop> {
private:
  explicit CFGLoop(mlir::Block *block);

  friend class llvm::LoopBase<mlir::Block, CFGLoop>;
  friend class llvm::LoopInfoBase<mlir::Block, CFGLoop>;
};

/// An LLVM LoopInfo instantiation for MLIR that provides access to CFG loops
/// found in the dominator tree.
class CFGLoopInfo : public llvm::LoopInfoBase<mlir::Block, mlir::CFGLoop> {
public:
  CFGLoopInfo(const llvm::DominatorTreeBase<mlir::Block, false> &domTree);
};

} // namespace mlir

#endif // MLIR_ANALYSIS_LOOPINFO_H
