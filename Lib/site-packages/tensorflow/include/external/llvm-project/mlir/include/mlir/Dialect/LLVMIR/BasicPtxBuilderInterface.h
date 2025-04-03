//===- BasicPtxBuilderInterface.td - PTX builder interface -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to build PTX (Parallel Thread Execution) from NVVM Ops
// automatically. It is used by NVVM to LLVM pass.
//
//===----------------------------------------------------------------------===//

#ifndef NVVM_DIALECT_NVVM_IR_BASICPTXBUILDERINTERFACE_H_
#define NVVM_DIALECT_NVVM_IR_BASICPTXBUILDERINTERFACE_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace NVVM {
/// Register read/write modifier to build constraint string for PTX inline
/// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#parameters
enum class PTXRegisterMod {
  /// Read register with no modifier
  Read = 0,
  /// Read register with '+' modifier
  Write = 2,
  /// Read register with '=' modifier.
  /// Note that, this is not natively supported by LLVM, but it is possible to
  /// set read and write for the same operand.
  ReadWrite = 1,
};
} // namespace NVVM
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Dialect/LLVMIR/BasicPtxBuilderInterface.h.inc"

namespace mlir {

namespace NVVM {

/// A class to build PTX assembly automatically. It is used by
/// BasicPtxBuilderInterface.
class PtxBuilder {
  // The interface op that is used to build the PTX.
  BasicPtxBuilderInterface interfaceOp;
  // Rewriter to create new operations.
  PatternRewriter &rewriter;
  // The operands for the PTX instruction
  SmallVector<Value> ptxOperands;
  // Register constraints (read, write, readwrite) and register data types
  std::string registerConstraints;

  bool hasResult = false;

public:
  /// Single constructor that only initializes members.
  PtxBuilder(Operation *op, PatternRewriter &rewriter)
      : interfaceOp(op), rewriter(rewriter) {}

  /// Add an operand with the read/write input type.
  void insertValue(Value v, PTXRegisterMod itype = PTXRegisterMod::Read);

  /// Builds the inline assembly Op and returns it. The `insertValue` needs to
  /// be called to pass operands before building the PTX.
  LLVM::InlineAsmOp build();

  /// Shortcut to build the inline assembly Op and replace or erase the original
  /// op with
  void buildAndReplaceOp();
};

} // namespace NVVM
} // namespace mlir

#endif // NVVM_DIALECT_NVVM_IR_BASICPTXBUILDERINTERFACE_H_
