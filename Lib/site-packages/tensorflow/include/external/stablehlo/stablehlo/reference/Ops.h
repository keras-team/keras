/* Copyright 2023 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef STABLEHLO_REFERENCE_OPS_H
#define STABLEHLO_REFERENCE_OPS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Axes.h"
#include "stablehlo/reference/Configuration.h"
#include "stablehlo/reference/Process.h"
#include "stablehlo/reference/ProcessGrid.h"
#include "stablehlo/reference/Scope.h"
#include "stablehlo/reference/Tensor.h"
#include "stablehlo/reference/Token.h"
#include "stablehlo/reference/Value.h"

namespace mlir {
namespace stablehlo {

// Evaluators for StableHLO ops.
Tensor absOp(const Tensor &operand, ShapedType resultType);
Tensor addOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Token afterAllOp(ArrayRef<Token> inputs, MLIRContext *context);
SmallVector<InterpreterValue> allGatherOp(
    ArrayRef<Tensor> operands, int64_t allGatherDim,
    SmallVector<SmallVector<uint32_t>> replicaGroups, ChannelId channelId,
    bool useGlobalDeviceIds, Process *process,
    ArrayRef<ShapedType> resultTypes);
SmallVector<InterpreterValue> allReduceOp(
    ArrayRef<Tensor> operands, SmallVector<SmallVector<uint32_t>> replicaGroups,
    ChannelId channelId, bool useGlobalDeviceIds, Region &computation,
    Process *process, Scope &scope, ArrayRef<ShapedType> resultTypes);
SmallVector<InterpreterValue> allToAllOp(
    ArrayRef<Tensor> operands, Axis splitDimension, Axis concatDimension,
    int64_t splitCount, SmallVector<SmallVector<uint32_t>> replicaGroups,
    ChannelId channelId, Process *process, ArrayRef<ShapedType> resultTypes);
Tensor andOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor atan2Op(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor bitcastConvertOp(const Tensor &operand, ShapedType resultType);
Tensor broadcastInDimOp(const Tensor &operand, const Axes &broadcastDimensions,
                        ShapedType resultType);
SmallVector<InterpreterValue> caseOp(const Tensor &index, RegionRange branches,
                                     Process *process, Scope &scope);
Tensor cbrtOp(const Tensor &operand, ShapedType resultType);
Tensor ceilOp(const Tensor &operand, ShapedType resultType);
Tensor clampOp(const Tensor &min, const Tensor &operand, const Tensor &max,
               ShapedType resultType);
Tensor clzOp(const Tensor &operand, ShapedType resultType);
Tensor collectiveBroadcastOp(const Tensor &operand,
                             SmallVector<SmallVector<uint32_t>> replicaGroups,
                             ChannelId channelId, Process *process);
Tensor collectivePermuteOp(const Tensor &operand,
                           SmallVector<SmallVector<uint32_t>> sourceTargetPairs,
                           ChannelId channelId, Process *process);
Tensor compareOp(const Tensor &lhs, const Tensor &rhs,
                 ComparisonDirection comparisonDirection,
                 ShapedType resultType);
Tensor complexOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor concatenateOp(ArrayRef<Tensor> inputs, Axis dimension,
                     ShapedType resultType);
Tensor constantOp(ElementsAttr value);
Tensor convertOp(const Tensor &operand, ShapedType resultType);
Tensor convolutionOp(
    const Tensor &lhs, const Tensor &rhs, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    ArrayRef<bool> windowReversal, Axis inputBatchDimension,
    Axis inputFeatureDimension, const Axes &inputSpatialDimensions,
    Axis kernelInputFeatureDimension, Axis kernelOutputFeatureDimension,
    const Axes &kernelSpatialDimensions, Axis outputBatchDimension,
    Axis outputFeatureDimension, const Axes &outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount, ShapedType resultType);
Tensor cosineOp(const Tensor &operand, ShapedType resultType);
Tensor divideOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor dotGeneralOp(const Tensor &lhs, const Tensor &rhs,
                    const Axes &lhsBatchingDimensions,
                    const Axes &rhsBatchingDimensions,
                    const Axes &lhsContractingDimensions,
                    const Axes &rhsContractingDimensions,
                    ShapedType resultType);
Tensor dynamicSliceOp(const Tensor &operand, ArrayRef<Tensor> startIndices,
                      const Sizes &sliceSizes, ShapedType resultType);
Tensor dynamicUpdateSliceOp(const Tensor &operand, const Tensor &update,
                            ArrayRef<Tensor> startIndices,
                            ShapedType resultType);
Tensor expm1Op(const Tensor &operand, ShapedType resultType);
Tensor exponentialOp(const Tensor &operand, ShapedType resultType);
Tensor floorOp(const Tensor &operand, ShapedType resultType);
Tensor gatherOp(const Tensor &operand, const Tensor &startIndices,
                const Axes &offsetDims, const Axes &collapsedSliceDims,
                const Axes &operandBatchingDims,
                const Axes &startIndicesBatchingDims, const Axes &startIndexMap,
                Axis indexVectorDim, const Sizes &sliceSizes,
                bool indicesAreSorted, ShapedType resultType);
Tensor getDimensionSizeOp(const Tensor &operand, Axis dimension,
                          ShapedType resultType);
InterpreterValue getTupleElementOp(const Tuple &operand, int32_t index);
SmallVector<InterpreterValue> ifOp(const Tensor &pred, Region &trueBranch,
                                   Region &falseBranch, Process *process,
                                   Scope &scope);
Tensor imagOp(const Tensor &operand, ShapedType resultType);
SmallVector<InterpreterValue> infeedOp(Token token, Process *process,
                                       Region &region, Scope &scope);
Tensor iotaOp(Axis iotaDimension, ShapedType resultType);
Tensor isFiniteOp(const Tensor &operand, ShapedType resultType);
Tensor log1pOp(const Tensor &operand, ShapedType resultType);
Tensor logOp(const Tensor &operand, ShapedType resultType);
Tensor logisticOp(const Tensor &operand, ShapedType resultType);
Tensor mapOp(ArrayRef<Tensor> inputs, Region &computation, Process *process,
             Scope &scope, ShapedType resultType);
Tensor maxOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor minOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor multiplyOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor negOp(const Tensor &operand, ShapedType resultType);
Tensor notOp(const Tensor &operand, ShapedType resultType);
SmallVector<InterpreterValue> optimizationBarrierOp(
    ArrayRef<InterpreterValue> operand);
Tensor orOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Token outfeedOp(ArrayRef<Tensor> inputs, Token token, Process *process);
Tensor padOp(const Tensor &operand, const Tensor &paddingValue,
             const Sizes &edgePaddingLow, const Sizes &interiorPadding,
             ShapedType resultType);
Tensor partitionIdOp(Process *process, MLIRContext *context);
Tensor populationCountOp(const Tensor &operand, ShapedType resultType);
Tensor powerOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor realOp(const Tensor &operand, ShapedType resultType);
SmallVector<InterpreterValue> recvOp(Token token, ChannelId channelId,
                                     Process *process);
SmallVector<Tensor> reduceOp(ArrayRef<Tensor> inputs,
                             ArrayRef<Tensor> initValues,
                             const Axes &dimensions, Region &body,
                             Process *process, Scope &scope,
                             ArrayRef<ShapedType> resultTypes);
Tensor reducePrecisionOp(const Tensor &operand, int32_t exponentBits,
                         int32_t mantissaBits, ShapedType resultType);
Tensor reduceScatterOp(const Tensor &operand, int64_t scatterDimension,
                       SmallVector<SmallVector<uint32_t>> replicaGroups,
                       ChannelId channelId, bool useGlobalDeviceIds,
                       Region &region, Process *process, Scope &scope,
                       ShapedType returnType);
SmallVector<Tensor> reduceWindowOp(
    ArrayRef<Tensor> inputs, ArrayRef<Tensor> initValues,
    const Sizes &windowDimensions, const Sizes &windowStrides,
    const Sizes &baseDilations, const Sizes &windowDilations,
    const Sizes &paddingLow, const Sizes &paddingHigh, Region &body,
    Process *process, Scope &scope, ArrayRef<ShapedType> resultTypes);
Tensor remOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor replicaIdOp(Process *process, MLIRContext *context);
Tensor reshapeOp(const Tensor &operand, ShapedType resultType);
Tensor reverseOp(const Tensor &operand, const Axes &dimensions,
                 ShapedType resultType);
Tensor roundOp(const Tensor &operand, ShapedType resultType);
Tensor roundNearestEvenOp(const Tensor &operand, ShapedType resultType);
Tensor rsqrtOp(const Tensor &operand, ShapedType resultType);
SmallVector<Tensor> scatterOp(
    ArrayRef<Tensor> inputs, const Tensor &scatterIndices,
    ArrayRef<Tensor> updates, const Axes &updateWindowDims,
    const Axes &insertedWindowDims, const Axes &inputBatchingDims,
    const Axes &scatterIndicesBatchingDims,
    const Axes &scatterDimsToOperandDims, Axis indexVectorDim,
    Region &updateComputation, Process *process, Scope &scope,
    ArrayRef<ShapedType> resultTypes);
Tensor selectOp(const Tensor &pred, const Tensor &onTrue, const Tensor &onFalse,
                ShapedType resultType);
Tensor selectAndScatterOp(const Tensor &operand, const Tensor &source,
                          const Tensor &initValue,
                          const Sizes &windowDimensions,
                          const Sizes &windowStrides, const Sizes &paddingLow,
                          Region &select, Region &scatter, Process *process,
                          Scope &scope, ShapedType resultType);
Token sendOp(ArrayRef<Tensor> inputs, Token token, ChannelId channelId,
             Process *process);
Tensor shiftLeftOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor shiftRightArithmeticOp(const Tensor &lhs, const Tensor &rhs,
                              ShapedType resultType);
Tensor shiftRightLogicalOp(const Tensor &lhs, const Tensor &rhs,
                           ShapedType resultType);
Tensor signOp(const Tensor &operand, ShapedType resultType);
Tensor sineOp(const Tensor &operand, ShapedType resultType);
Tensor sliceOp(const Tensor &operand, const Sizes &startIndices,
               const Sizes &strides, ShapedType resultType);
SmallVector<Tensor> sortOp(ArrayRef<Tensor> inputs, Axis dimension,
                           bool isStable, Region &comparator, Process *process,
                           Scope &scope);
Tensor sqrtOp(const Tensor &operand, ShapedType resultType);
Tensor subtractOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);
Tensor tanOp(const Tensor &operand, ShapedType resultType);
Tensor tanhOp(const Tensor &operand, ShapedType resultType);
Tensor transposeOp(const Tensor &operand, const Axes &permutation,
                   ShapedType resultType);
Tuple tupleOp(ArrayRef<InterpreterValue> val, TupleType resultType);
SmallVector<InterpreterValue> whileOp(SmallVector<InterpreterValue> operand,
                                      Region &cond, Region &body,
                                      InterpreterFallback *fallback,
                                      Process *process, Scope &scope);
Tensor xorOp(const Tensor &lhs, const Tensor &rhs, ShapedType resultType);

/// Evaluates an mlir::Region `region` using the runtime values `args`
/// corresponding to the arguments of the entry block of the region.
/// Interprets the operations within the entry block and returns the runtime
/// values for the terminator's arguments. The optional callback `fallback` is
/// used for evaluating ops which are not supported by the interpreter.
/// Assumes that `region` has only one block.
SmallVector<InterpreterValue> eval(Region &region,
                                   ArrayRef<InterpreterValue> args,
                                   InterpreterFallback *fallback = nullptr,
                                   Process *process = nullptr,
                                   Scope *parent = nullptr);

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_REFERENCE_OPS_H
