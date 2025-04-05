//===- DstBufferizableOpInterfaceImpl.h - Dst Op Bufferization --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_DSTBUFFERIZABLEOPINTERFACEIMPL_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_DSTBUFFERIZABLEOPINTERFACEIMPL_H_

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {
namespace bufferization {

/// Bufferizable ops that implement the DestinationStyleOpInterface can use this
/// external model base class. It provides default implementations for various
/// required interface methods.
template <typename ConcreteModel, typename ConcreteOp>
struct DstBufferizableOpInterfaceExternalModel
    : public BufferizableOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // All inputs and outputs bufferize to a memory read.
    assert(isa<DestinationStyleOpInterface>(op) &&
           "expected that op implements DestinationStyleOpInterface");
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Only outputs bufferize to a memory write.
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    return dstOp.isDpsInit(&opOperand);
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // Output operands alias with their respective tied OpResults.
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.isDpsInit(&opOperand))
      return {{dstOp.getTiedOpResult(&opOperand), BufferRelation::Equivalent}};
    return {};
  }
};

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_IR_DSTBUFFERIZABLEOPINTERFACEIMPL_H_
