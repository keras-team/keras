//===- VectorToSCF.h - Convert vector to SCF dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOSCF_VECTORTOSCF_H_
#define MLIR_CONVERSION_VECTORTOSCF_VECTORTOSCF_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir {
class MLIRContext;
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTVECTORTOSCF
#include "mlir/Conversion/Passes.h.inc"

/// When lowering an N-d vector transfer op to an (N-1)-d vector transfer op,
/// a temporary buffer is created through which individual (N-1)-d vector are
/// staged. This pattern can be applied multiple time, until the transfer op
/// is 1-d.
/// This is consistent with the lack of an LLVM instruction to dynamically
/// index into an aggregate (see the Vector dialect lowering to LLVM deep dive).
///
/// An instruction such as:
/// ```
///    vector.transfer_write %vec, %A[%a, %b, %c] :
///      vector<9x17x15xf32>, memref<?x?x?xf32>
/// ```
/// Lowers to pseudo-IR resembling (unpacking one dimension):
/// ```
///    %0 = alloca() : memref<vector<9x17x15xf32>>
///    store %vec, %0[] : memref<vector<9x17x15xf32>>
///    %1 = vector.type_cast %0 :
///      memref<vector<9x17x15xf32>> to memref<9xvector<17x15xf32>>
///    affine.for %I = 0 to 9 {
///      %dim = dim %A, 0 : memref<?x?x?xf32>
///      %add = affine.apply %I + %a
///      %cmp = arith.cmpi "slt", %add, %dim : index
///      scf.if %cmp {
///        %vec_2d = load %1[%I] : memref<9xvector<17x15xf32>>
///        vector.transfer_write %vec_2d, %A[%add, %b, %c] :
///          vector<17x15xf32>, memref<?x?x?xf32>
/// ```
///
/// When applying the pattern a second time, the existing alloca() operation
/// is reused and only a second vector.type_cast is added.
struct VectorTransferToSCFOptions {
  /// Minimal rank to which vector transfer are lowered.
  unsigned targetRank = 1;
  VectorTransferToSCFOptions &setTargetRank(unsigned r) {
    targetRank = r;
    return *this;
  }
  /// Allows vector transfers that operated on tensors to be lowered (this is an
  /// uncommon alternative).
  bool lowerTensors = false;
  VectorTransferToSCFOptions &enableLowerTensors(bool l = true) {
    lowerTensors = l;
    return *this;
  }
  /// Triggers full unrolling (vs iterating with a loop) during transfer to scf.
  bool unroll = false;
  VectorTransferToSCFOptions &enableFullUnroll(bool u = true) {
    unroll = u;
    return *this;
  }
  /// Enable scalable vector specific lowerings (which introduce loops). These
  /// work alongside fullUnroll (which unrolls until the first scalable
  /// dimension).
  bool lowerScalable = false;
  VectorTransferToSCFOptions enableLowerScalable(bool enable = true) {
    lowerScalable = enable;
    return *this;
  }
};

/// Collect a set of patterns to convert from the Vector dialect to SCF + func.
void populateVectorToSCFConversionPatterns(
    RewritePatternSet &patterns,
    const VectorTransferToSCFOptions &options = VectorTransferToSCFOptions());

/// Create a pass to convert a subset of vector ops to SCF.
std::unique_ptr<Pass> createConvertVectorToSCFPass(
    const VectorTransferToSCFOptions &options = VectorTransferToSCFOptions());

} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOSCF_VECTORTOSCF_H_
