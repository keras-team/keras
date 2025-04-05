/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H
#define STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H

#include <optional>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"

namespace mlir {
namespace hlo {

//===----------------------------------------------------------------------===//
// Utilities for shape functions
//===----------------------------------------------------------------------===//

void reifyGatherDimSizes(int64_t resultRank,
                         llvm::function_ref<Value(int64_t)> getStartIndicesDim,
                         llvm::function_ref<Value(int64_t)> getSliceDim,
                         ArrayRef<int64_t> offsetDims,
                         ArrayRef<int64_t> collapsedSliceDims,
                         ArrayRef<int64_t> operandBatchingDims,
                         int64_t indexVectorDim, SmallVectorImpl<Value>& shape);

// Convert a Nx2 dense int64 padding attribute to a list of tuples.
FailureOr<SmallVector<std::pair<int64_t, int64_t>>> convertPaddingAttribute(
    std::optional<DenseIntElementsAttr> optionalAttr,
    std::optional<Location> loc);

// Convert a 1D dense bool attribute to a list of values.
FailureOr<SmallVector<bool>> convertWindowReversalAttribute(
    std::optional<DenseElementsAttr> optionalAttr, std::optional<Location> loc,
    StringRef attrName);

// WindowDimension described how the kernel window moves across the base area
// in a particular dimension.
// Describes the windowing in an operation such as convolution.
// The window is moved across a base area and for each position of the
// window a computation is performed. The field below describes the
// window and the movement of the window across a base area.
struct WindowDimension {
  int64_t size = 0;
  int64_t stride = 1;
  int64_t paddingLow = 0;
  int64_t paddingHigh = 0;
  int64_t windowDilation = 1;
  int64_t baseDilation = 1;
  bool windowReversal = false;
};

FailureOr<SmallVector<WindowDimension>>
verifyWindowAttributesAndInferWindowDimensions(
    ArrayRef<int64_t> windowDimensions, ArrayRef<int64_t> windowStrides,
    ArrayRef<std::pair<int64_t, int64_t>> padding,
    ArrayRef<int64_t> lhsDilation, ArrayRef<int64_t> rhsDilation,
    ArrayRef<bool> windowReversal, std::optional<Location> loc);

SmallVector<int64_t> inferWindowOutputShape(ArrayRef<int64_t> baseShape,
                                            ArrayRef<WindowDimension> window);

LogicalResult verifyReplicaGroups(std::optional<Location> location,
                                  DenseIntElementsAttr replicaGroups,
                                  bool allGroupsMustHaveSameSize,
                                  bool useGlobalDeviceIds,
                                  std::optional<size_t> expectedGroupSize);

LogicalResult verifyConvolutionAttributes(
    std::optional<Location> location, Type lhsType, Type rhsType,
    int64_t inputBatchDimension, int64_t inputFeatureDimension,
    ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    std::optional<ArrayAttr> precisionConfig);

//===----------------------------------------------------------------------===//
// Shape functions for ops.
//===----------------------------------------------------------------------===//
// These functions have been moved out of StablehloOps.cpp in order to be
// shared with the MHLO dialect.
// Because of that, they cannot use any definitions in the StableHLO dialect
// (definitions in Base are fine, because they are shared with MHLO).
// As a result, these definitions (e.g. StableHLO ops and attributes) are
// decomposed into smaller pieces which are passed as individual parameters.
// These parameters have the same names as in the ODS and come in the same
// order in which they are declared in the ODS.

LogicalResult inferAbsOp(std::optional<Location>, Value operand,
                         SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferAfterAllOp(HloDialectInterface* dialect,
                              std::optional<Location> location,
                              SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferAllToAllOp(
    std::optional<Location> location, ValueRange operands,
    int64_t splitDimension, int64_t concatDimension, int64_t splitCount,
    DenseIntElementsAttr replicaGroups,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferAllReduceOp(
    std::optional<Location> location, ValueRange operands, Region& computation,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormGradOp(
    std::optional<Location> location, Value operand, Value scale, Value mean,
    Value variance, Value gradOutput, int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormInferenceOp(
    std::optional<Location> location, Value operand, Value scale, Value offset,
    Value mean, Value variance, int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBatchNormTrainingOp(
    std::optional<Location> location, Value operand, Value scale, Value offset,
    int64_t featureIndex,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferBroadcastOp(
    std::optional<Location> location, Value operand,
    ArrayRef<int64_t> broadcastSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferCaseOp(std::optional<Location> location, Value index,
                          RegionRange branches,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferCholeskyOp(
    std::optional<Location> location, Value a,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferClampOp(
    std::optional<Location> location, Value min, Value operand, Value max,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferCollectiveBroadcastOp(
    std::optional<Location>, ValueRange operands,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferCollectivePermuteOp(
    std::optional<Location>, ValueRange operands,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferCompareOp(
    MLIRContext* context, std::optional<Location>, Value lhs,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferComplexOp(std::optional<Location> location, Value lhs,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferConcatenateOp(std::optional<Location> location,
                                 TypeRange inputTypes, int64_t dimension,
                                 SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferConstantOp(std::optional<Location>, ElementsAttr value,
                              SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferConvertOp(
    std::optional<Location> location, Value operand,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferConvolutionOp(
    std::optional<Location> location, Type lhsType, Type rhsType,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<DenseIntElementsAttr> padding,
    std::optional<ArrayRef<int64_t>> lhsDilation,
    std::optional<ArrayRef<int64_t>> rhsDilation,
    std::optional<ArrayRef<bool>> windowReversal, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    std::optional<ArrayAttr> precisionConfig,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferCreateTokenOp(HloDialectInterface* dialect,
                                 std::optional<Location> location,
                                 SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferDotOp(
    std::optional<Location> location, RankedTensorType lhsType,
    RankedTensorType rhsType, std::optional<ArrayAttr> precisionConfig,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDotGeneralOp(
    std::optional<Location> location, Type lhsType, Type rhsType,
    ArrayRef<int64_t> lhsBatchingDimensions,
    ArrayRef<int64_t> rhsBatchingDimensions,
    ArrayRef<int64_t> lhsContractingDimensions,
    ArrayRef<int64_t> rhsContractingDimensions,
    std::optional<ArrayAttr> precisionConfig,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDynamicConvOp(
    std::optional<Location> location, Type lhsType, Type rhsType, Value padding,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<ArrayRef<int64_t>> lhsDilation,
    std::optional<ArrayRef<int64_t>> rhsDilation,
    std::optional<ArrayRef<bool>> windowReversal, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    std::optional<ArrayAttr> precisionConfig,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDynamicGatherOp(
    std::optional<Location> location, Value operand, Value startIndices,
    Value sliceSizes, ArrayRef<int64_t> offsetDims,
    ArrayRef<int64_t> collapsedSliceDims, ArrayRef<int64_t> operandBatchingDims,
    ArrayRef<int64_t> startIndicesBatchingDims, ArrayRef<int64_t> startIndexMap,
    int64_t indexVectorDim,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDynamicSliceOp(
    std::optional<Location> location, Type operandType,
    TypeRange startIndicesTypes, ArrayRef<int64_t> sliceSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferDynamicUpdateSliceOp(
    std::optional<Location> location, Value operand, Value update,
    ValueRange startIndices,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferFftOp(
    std::optional<Location> location, Value operand, bool isFftTypeRfft,
    bool isFftTypeIrfft, ArrayRef<int64_t> fftLength,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferGatherOp(
    std::optional<Location> location, Value operand, Value startIndices,
    ArrayRef<int64_t> offsetDims, ArrayRef<int64_t> collapsedSliceDims,
    ArrayRef<int64_t> operandBatchingDims,
    ArrayRef<int64_t> startIndicesBatchingDims, ArrayRef<int64_t> startIndexMap,
    int64_t indexVectorDim, ArrayRef<int64_t> sliceSizes,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferGetDimensionSizeOp(
    std::optional<Location> location, Type operandType, int64_t dimension,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferGetTupleElementOp(
    std::optional<Location> location, Value operand, int32_t index,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferImagOp(std::optional<Location> location, Value operand,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferIsFiniteOp(MLIRContext* context, std::optional<Location>,
                              Value x,
                              SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferIfOp(std::optional<Location> location, Value pred,
                        RegionRange branches,
                        SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferMapOp(
    std::optional<Location> location, ValueRange inputs,
    ArrayRef<int64_t> dimensions, Region& computation,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferOptimizationBarrierOp(
    std::optional<Location> location, ValueRange operand,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferOutfeedOp(HloDialectInterface* dialect,
                             std::optional<Location> location,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferPadOp(std::optional<Location> location, Type operandType,
                         Type paddingValueType,
                         ArrayRef<int64_t> edgePaddingLow,
                         ArrayRef<int64_t> edgePaddingHigh,
                         ArrayRef<int64_t> interiorPadding,
                         SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferPartitionIdOp(MLIRContext* context,
                                 std::optional<Location> location,
                                 SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferRealOp(std::optional<Location> location, Value operand,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferReduceOp(
    std::optional<Location> location, TypeRange inputTypes,
    ArrayRef<int64_t> dimensions, Region& body,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReduceWindowOp(
    std::optional<Location> location, ValueRange inputs, ValueRange initValues,
    ArrayRef<int64_t> windowDimensions,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<ArrayRef<int64_t>> baseDilations,
    std::optional<ArrayRef<int64_t>> windowDilations,
    std::optional<DenseIntElementsAttr> padding, Region& body,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferReplicaIdOp(MLIRContext* context, std::optional<Location>,
                               SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferReverseOp(std::optional<Location> location, Type operandType,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferRngOp(
    std::optional<Location> location, Value a, Value b, Value shape,
    bool isRngDistributionUniform,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferScatterOp(std::optional<Location> location,
                             ValueRange inputs, Region& update_computation,
                             SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSelectOp(
    std::optional<Location> location, Value pred, Value onTrue, Value onFalse,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferSelectAndScatterOp(
    std::optional<Location> location, Value operand, Region& scatter,
    SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSendOp(HloDialectInterface* dialect,
                          std::optional<Location> location,
                          bool isDeviceToDevice, bool isDeviceToHost,
                          bool isHostTransfer,
                          SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSetDimensionSizeOp(
    HloDialectInterface* dialect, std::optional<Location> location,
    Type operandType, Value size, int64_t dimension,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferSliceOp(std::optional<Location> location, Type operandType,
                           ArrayRef<int64_t> startIndices,
                           ArrayRef<int64_t> limitIndices,
                           ArrayRef<int64_t> strides,
                           SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferSortOp(
    std::optional<Location> location, ValueRange inputs,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferTopKOp(
    std::optional<Location> location, Value operand, int64_t k,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferTransposeOp(std::optional<Location> loc, Value operand,
                               ArrayRef<int64_t> permutation,
                               SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferTriangularSolveOp(
    std::optional<Location> location, Value a, Value b, bool leftSide,
    bool isTransposeAInvalid,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferTupleOp(MLIRContext* context,
                           std::optional<Location> location, ValueRange val,
                           SmallVectorImpl<Type>& inferredReturnTypes);

LogicalResult inferUniformDequantizeOp(
    std::optional<Location> location, Value operand,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferUniformQuantizeOp(
    std::optional<Location> location, Value operand,
    SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes);

LogicalResult inferWhileOp(std::optional<Location> location, ValueRange operand,
                           SmallVectorImpl<Type>& inferredReturnTypes);

//===----------------------------------------------------------------------===//
// Verifiers for ops.
//===----------------------------------------------------------------------===//

LogicalResult verifyAddOp(std::optional<Location> location, Operation* op,
                          Type lhsType, Type rhsType, Type resultType);

LogicalResult verifyAllGatherOp(std::optional<Location> location,
                                ValueRange operands, int64_t allGatherDim,
                                DenseIntElementsAttr replicaGroups,
                                int64_t channelId, bool useGlobalDeviceIds,
                                ValueRange results);

LogicalResult verifyAllReduceOp(std::optional<Location> location,
                                ValueRange operands,
                                DenseIntElementsAttr replicaGroups,
                                int64_t channelId, bool useGlobalDeviceIds,
                                Region& computation);

LogicalResult verifyBitcastConvertOp(std::optional<Location> location,
                                     Value operand, Value result);

LogicalResult verifyBroadcastInDimOp(std::optional<Location> location,
                                     Value operand,
                                     ArrayRef<int64_t> broadcastDimensions,
                                     Value result);

LogicalResult verifyCollectiveBroadcastOp(std::optional<Location> location,
                                          DenseIntElementsAttr replicaGroups);

LogicalResult verifyCollectivePermuteOp(std::optional<Location> location,
                                        DenseIntElementsAttr sourceTargetPairs);

LogicalResult verifyCompositeOp(std::optional<Location> location, Operation* op,
                                StringRef name, StringRef decomposition,
                                SymbolTableCollection& symbolTable);

LogicalResult verifyConvolutionOp(
    std::optional<Location> location, Type lhsType, Type rhsType,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<DenseIntElementsAttr> padding,
    std::optional<ArrayRef<int64_t>> lhsDilation,
    std::optional<ArrayRef<int64_t>> rhsDilation,
    std::optional<ArrayRef<bool>> windowReversal, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    std::optional<ArrayAttr> precisionConfig, Type resultType);

LogicalResult verifyDotOp(std::optional<Location> location,
                          RankedTensorType lhsType, RankedTensorType rhsType,
                          std::optional<ArrayAttr> precisionConfig,
                          Value result);

LogicalResult verifyDotGeneralOp(std::optional<Location> location, Value lhs,
                                 Value rhs,
                                 ArrayRef<int64_t> lhsBatchingDimensions,
                                 ArrayRef<int64_t> rhsBatchingDimensions,
                                 ArrayRef<int64_t> lhsContractingDimensions,
                                 ArrayRef<int64_t> rhsContractingDimensions,
                                 std::optional<ArrayAttr> precisionConfig,
                                 bool isDefaultPrecisionConfig,
                                 bool hasAlgorithmSpecified, Value result);

LogicalResult verifyDotAlgorithmAttr(
    ::llvm::function_ref<InFlightDiagnostic()> emitError, Type lhsPrecisionType,
    Type rhsPrecisionType, Type accumulationType, int64_t lhsComponentCount,
    int64_t rhsComponentCount, int64_t numPrimitiveOperations,
    bool allowImpreciseAccumulation);

LogicalResult verifyDynamicBroadcastInDimOp(
    std::optional<Location> location, Value operand, Value outputDimensions,
    ArrayRef<int64_t> broadcastDimensions,
    std::optional<ArrayRef<int64_t>> knownExpandingDimensions,
    std::optional<ArrayRef<int64_t>> knownNonexpandingDimensions, Value result);

LogicalResult verifyDynamicConvOp(
    std::optional<Location> location, Type lhsType, Type rhsType, Value padding,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<ArrayRef<int64_t>> lhsDilation,
    std::optional<ArrayRef<int64_t>> rhsDilation,
    std::optional<ArrayRef<bool>> windowReversal, int64_t inputBatchDimension,
    int64_t inputFeatureDimension, ArrayRef<int64_t> inputSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    ArrayRef<int64_t> kernelSpatialDimensions, int64_t outputBatchDimension,
    int64_t outputFeatureDimension, ArrayRef<int64_t> outputSpatialDimensions,
    int64_t featureGroupCount, int64_t batchGroupCount,
    std::optional<ArrayAttr> precisionConfig, Type resultType);

LogicalResult verifyDynamicIotaOp(std::optional<Location> location,
                                  Value outputShape, int64_t iotaDimension,
                                  Value result);

LogicalResult verifyDynamicPadOp(std::optional<Location> location,
                                 Value operand, Value paddingValue,
                                 Value edgePaddingLow, Value edgePaddingHigh,
                                 Value interiorPadding, Value result);

LogicalResult verifyDynamicReshapeOp(std::optional<Location> location,
                                     Value operand, Value outputShape,
                                     Value result);

LogicalResult verifyInfeedOp(HloDialectInterface* dialect,
                             std::optional<Location> location,
                             std::optional<ArrayAttr> layout,
                             ValueRange results);

LogicalResult verifyIotaOp(std::optional<Location> location,
                           int64_t iotaDimension, Value result);

LogicalResult verifyRealDynamicSliceOp(std::optional<Location> location,
                                       Value operand, Value startIndices,
                                       Value limitIndices, Value strides);

LogicalResult verifyRecvOp(HloDialectInterface* dialect,
                           std::optional<Location> location,
                           bool isDeviceToDevice, bool isHostToDevice,
                           bool isHostTransfer, ValueRange results);

LogicalResult verifyReduceOp(std::optional<Location> location,
                             ValueRange inputs, ValueRange initValues,
                             ArrayRef<int64_t> dimensions, Region& body);

LogicalResult verifyReduceOpInputsAndInferShape(
    std::optional<Location> location, SmallVector<ShapedType> inputTypes,
    ArrayRef<int64_t> dimensions, SmallVector<int64_t>& newDimensions,
    Attribute& encoding);

LogicalResult verifyReduceScatterOp(std::optional<Location> location,
                                    Value operand, int64_t scatterDimension,
                                    DenseIntElementsAttr replicaGroups,
                                    int64_t channelId, bool useGlobalDeviceIds,
                                    Region& computation, Value result);

LogicalResult verifyReduceWindowOp(
    std::optional<Location> location, ValueRange inputs, ValueRange initValues,
    ArrayRef<int64_t> windowDimensions,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<ArrayRef<int64_t>> baseDilations,
    std::optional<ArrayRef<int64_t>> windowDilations,
    std::optional<DenseIntElementsAttr> padding, Region& body);

LogicalResult verifyReshapeOp(std::optional<Location> location, Value operand,
                              Value result);

LogicalResult verifyReverseOp(std::optional<Location> location, Value operand,
                              llvm::ArrayRef<int64_t> dimensions);

LogicalResult verifyRngBitGeneratorOp(std::optional<Location> location,
                                      Value initialState, Value outputState);

LogicalResult verifyScatterOp(
    std::optional<Location> location, ValueRange inputs, Value scatterIndices,
    ValueRange updates, ArrayRef<int64_t> updateWindowDims,
    ArrayRef<int64_t> insertedWindowDims, ArrayRef<int64_t> inputBatchingDims,
    ArrayRef<int64_t> scatterIndicesBatchingDims,
    ArrayRef<int64_t> scatterDimsToOperandDims, int64_t indexVectorDim,
    Region& updateComputation);

LogicalResult verifySelectAndScatterOp(
    std::optional<Location> location, Value operand, Value source,
    Value initValue, std::optional<ArrayRef<int64_t>> windowDimensions,
    std::optional<ArrayRef<int64_t>> windowStrides,
    std::optional<DenseIntElementsAttr> padding, Region& select,
    Region& scatter);

LogicalResult verifySortOp(std::optional<Location> location, ValueRange inputs,
                           int64_t dimension, Region& comparator);

LogicalResult verifyTransposeOp(std::optional<Location> location,
                                Type operandType, ArrayRef<int64_t> permutation,
                                Type resultType);

LogicalResult verifyUniformQuantizeOp(std::optional<Location> location,
                                      Value operand, Value result);

LogicalResult verifyWhileOp(std::optional<Location> location,
                            ValueRange operand, Region& cond, Region& body);
}  // end namespace hlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_STABLEHLO_TYPE_INFERENCE_H
