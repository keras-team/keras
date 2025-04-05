//===- LoweringOptions.h - Common config for lowering to LLVM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a configuration shared by several conversions targeting the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_LOWERINGOPTIONS_H
#define MLIR_CONVERSION_LLVMCOMMON_LOWERINGOPTIONS_H

#include "llvm/IR/DataLayout.h"

namespace mlir {

class DataLayout;
class MLIRContext;

/// Value to pass as bitwidth for the index type when the converter is expected
/// to derive the bitwidth from the LLVM data layout.
static constexpr unsigned kDeriveIndexBitwidthFromDataLayout = 0;

/// Options to control the LLVM lowering. The struct is used to share lowering
/// options between passes, patterns, and type converter.
class LowerToLLVMOptions {
public:
  explicit LowerToLLVMOptions(MLIRContext *ctx);
  LowerToLLVMOptions(MLIRContext *ctx, const DataLayout &dl);

  bool useBarePtrCallConv = false;

  enum class AllocLowering {
    /// Use malloc for heap allocations.
    Malloc,

    /// Use aligned_alloc for heap allocations.
    AlignedAlloc,

    /// Do not lower heap allocations. Users must provide their own patterns for
    /// AllocOp and DeallocOp lowering.
    None
  };

  AllocLowering allocLowering = AllocLowering::Malloc;

  bool useGenericFunctions = false;

  /// The data layout of the module to produce. This must be consistent with the
  /// data layout used in the upper levels of the lowering pipeline.
  // TODO: this should be replaced by MLIR data layout when one exists.
  llvm::DataLayout dataLayout = llvm::DataLayout("");

  /// Set the index bitwidth to the given value.
  void overrideIndexBitwidth(unsigned bitwidth) {
    assert(bitwidth != kDeriveIndexBitwidthFromDataLayout &&
           "can only override to a concrete bitwidth");
    indexBitwidth = bitwidth;
  }

  /// Get the index bitwidth.
  unsigned getIndexBitwidth() const { return indexBitwidth; }

private:
  unsigned indexBitwidth;
};

} // namespace mlir

#endif // MLIR_CONVERSION_LLVMCOMMON_LOWERINGOPTIONS_H
