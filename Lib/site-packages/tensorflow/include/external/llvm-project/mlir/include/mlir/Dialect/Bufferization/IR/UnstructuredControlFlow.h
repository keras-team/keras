//===- UnstructuredControlFlow.h - Op Interface Helpers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_UNSTRUCTUREDCONTROLFLOW_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_UNSTRUCTUREDCONTROLFLOW_H_

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

//===----------------------------------------------------------------------===//
// Helpers for Unstructured Control Flow
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bufferization {

namespace detail {
/// Return a list of operands that are forwarded to the given block argument.
/// I.e., find all predecessors of the block argument's owner and gather the
/// operands that are equivalent to the block argument.
SmallVector<OpOperand *> getCallerOpOperands(BlockArgument bbArg);
} // namespace detail

/// A template that provides a default implementation of `getAliasingOpOperands`
/// for ops that support unstructured control flow within their regions.
template <typename ConcreteModel, typename ConcreteOp>
struct OpWithUnstructuredControlFlowBufferizableOpInterfaceExternalModel
    : public BufferizableOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    // Note: The user may want to override this function for OpResults in
    // case the bufferized result type is different from the bufferized type of
    // the aliasing OpOperand (if any).
    if (isa<OpResult>(value))
      return bufferization::detail::defaultGetBufferType(value, options,
                                                         invocationStack);

    // Compute the buffer type of the block argument by computing the bufferized
    // operand types of all forwarded values. If these are all the same type,
    // take that type. Otherwise, take only the memory space and fall back to a
    // buffer type with a fully dynamic layout map.
    BaseMemRefType bufferType;
    auto tensorType = cast<TensorType>(value.getType());
    for (OpOperand *opOperand :
         detail::getCallerOpOperands(cast<BlockArgument>(value))) {

      // If the forwarded operand is already on the invocation stack, we ran
      // into a loop and this operand cannot be used to compute the bufferized
      // type.
      if (llvm::find(invocationStack, opOperand->get()) !=
          invocationStack.end())
        continue;

      // Compute the bufferized type of the forwarded operand.
      BaseMemRefType callerType;
      if (auto memrefType =
              dyn_cast<BaseMemRefType>(opOperand->get().getType())) {
        // The operand was already bufferized. Take its type directly.
        callerType = memrefType;
      } else {
        FailureOr<BaseMemRefType> maybeCallerType =
            bufferization::getBufferType(opOperand->get(), options,
                                         invocationStack);
        if (failed(maybeCallerType))
          return failure();
        callerType = *maybeCallerType;
      }

      if (!bufferType) {
        // This is the first buffer type that we computed.
        bufferType = callerType;
        continue;
      }

      if (bufferType == callerType)
        continue;

        // If the computed buffer type does not match the computed buffer type
        // of the earlier forwarded operands, fall back to a buffer type with a
        // fully dynamic layout map.
#ifndef NDEBUG
      if (auto rankedTensorType = dyn_cast<RankedTensorType>(tensorType)) {
        assert(bufferType.hasRank() && callerType.hasRank() &&
               "expected ranked memrefs");
        assert(llvm::all_equal({bufferType.getShape(), callerType.getShape(),
                                rankedTensorType.getShape()}) &&
               "expected same shape");
      } else {
        assert(!bufferType.hasRank() && !callerType.hasRank() &&
               "expected unranked memrefs");
      }
#endif // NDEBUG

      if (bufferType.getMemorySpace() != callerType.getMemorySpace())
        return op->emitOpError("incoming operands of block argument have "
                               "inconsistent memory spaces");

      bufferType = getMemRefTypeWithFullyDynamicLayout(
          tensorType, bufferType.getMemorySpace());
    }

    if (!bufferType)
      return op->emitOpError("could not infer buffer type of block argument");

    return bufferType;
  }

protected:
  /// Assuming that `bbArg` is a block argument of a block that belongs to the
  /// given `op`, return all OpOperands of users of this block that are
  /// aliasing with the given block argument.
  AliasingOpOperandList
  getAliasingBranchOpOperands(Operation *op, BlockArgument bbArg,
                              const AnalysisState &state) const {
    assert(bbArg.getOwner()->getParentOp() == op && "invalid bbArg");

    // Gather aliasing OpOperands of all operations (callers) that link to
    // this block.
    AliasingOpOperandList result;
    for (OpOperand *opOperand : detail::getCallerOpOperands(bbArg))
      result.addAlias(
          {opOperand, BufferRelation::Equivalent, /*isDefinite=*/false});

    return result;
  }
};

/// A template that provides a default implementation of `getAliasingValues`
/// for ops that implement the `BranchOpInterface`.
template <typename ConcreteModel, typename ConcreteOp>
struct BranchOpBufferizableOpInterfaceExternalModel
    : public BufferizableOpInterface::ExternalModel<ConcreteModel, ConcreteOp> {
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    AliasingValueList result;
    auto branchOp = cast<BranchOpInterface>(op);
    auto operandNumber = opOperand.getOperandNumber();

    // Gather aliasing block arguments of blocks to which this op may branch to.
    for (const auto &it : llvm::enumerate(op->getSuccessors())) {
      Block *block = it.value();
      SuccessorOperands operands = branchOp.getSuccessorOperands(it.index());
      assert(operands.getProducedOperandCount() == 0 &&
             "produced operands not supported");
      if (operands.getForwardedOperands().empty())
        continue;
      // The first and last operands that are forwarded to this successor.
      int64_t firstOperandIndex =
          operands.getForwardedOperands().getBeginOperandIndex();
      int64_t lastOperandIndex =
          firstOperandIndex + operands.getForwardedOperands().size();
      bool matchingDestination = operandNumber >= firstOperandIndex &&
                                 operandNumber < lastOperandIndex;
      // A branch op may have multiple successors. Find the ones that correspond
      // to this OpOperand. (There is usually only one.)
      if (!matchingDestination)
        continue;
      // Compute the matching block argument of the destination block.
      BlockArgument bbArg =
          block->getArgument(operandNumber - firstOperandIndex);
      result.addAlias(
          {bbArg, BufferRelation::Equivalent, /*isDefinite=*/false});
    }

    return result;
  }
};

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_IR_UNSTRUCTUREDCONTROLFLOW_H_
