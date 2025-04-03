// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

_OPENVINO_OP_REG(Abs, ov::op::v0)
_OPENVINO_OP_REG(Acos, ov::op::v0)
_OPENVINO_OP_REG(Add, ov::op::v1)
_OPENVINO_OP_REG(Asin, ov::op::v0)
_OPENVINO_OP_REG(Atan, ov::op::v0)
_OPENVINO_OP_REG(AvgPool, ov::op::v1)
_OPENVINO_OP_REG(BatchNormInference, ov::op::v0)
_OPENVINO_OP_REG(BinaryConvolution, ov::op::v1)
_OPENVINO_OP_REG(Broadcast, ov::op::v1)
_OPENVINO_OP_REG(CTCGreedyDecoder, ov::op::v0)
_OPENVINO_OP_REG(Ceiling, ov::op::v0)
_OPENVINO_OP_REG(Clamp, ov::op::v0)
_OPENVINO_OP_REG(Concat, ov::op::v0)
_OPENVINO_OP_REG(Constant, ov::op::v0)
_OPENVINO_OP_REG(Convert, ov::op::v0)
_OPENVINO_OP_REG(ConvertLike, ov::op::v1)
_OPENVINO_OP_REG(Convolution, ov::op::v1)
_OPENVINO_OP_REG(ConvolutionBackpropData, ov::op::v1)
_OPENVINO_OP_REG(Cos, ov::op::v0)
_OPENVINO_OP_REG(Cosh, ov::op::v0)
_OPENVINO_OP_REG(DeformableConvolution, ov::op::v1)
_OPENVINO_OP_REG(DeformablePSROIPooling, ov::op::v1)
_OPENVINO_OP_REG(DepthToSpace, ov::op::v0)
_OPENVINO_OP_REG(DetectionOutput, ov::op::v0)
_OPENVINO_OP_REG(Divide, ov::op::v1)
_OPENVINO_OP_REG(Elu, ov::op::v0)
_OPENVINO_OP_REG(Erf, ov::op::v0)
_OPENVINO_OP_REG(Equal, ov::op::v1)
_OPENVINO_OP_REG(Exp, ov::op::v0)
_OPENVINO_OP_REG(FakeQuantize, ov::op::v0)
_OPENVINO_OP_REG(Floor, ov::op::v0)
_OPENVINO_OP_REG(FloorMod, ov::op::v1)
_OPENVINO_OP_REG(Gather, ov::op::v1)
_OPENVINO_OP_REG(GatherTree, ov::op::v1)
_OPENVINO_OP_REG(Greater, ov::op::v1)
_OPENVINO_OP_REG(GreaterEqual, ov::op::v1)
_OPENVINO_OP_REG(GroupConvolution, ov::op::v1)
_OPENVINO_OP_REG(GroupConvolutionBackpropData, ov::op::v1)
_OPENVINO_OP_REG(GRN, ov::op::v0)
_OPENVINO_OP_REG(HardSigmoid, ov::op::v0)
_OPENVINO_OP_REG(Interpolate, ov::op::v0)
_OPENVINO_OP_REG(Less, ov::op::v1)
_OPENVINO_OP_REG(LessEqual, ov::op::v1)
_OPENVINO_OP_REG(Log, ov::op::v0)
_OPENVINO_OP_REG(LogicalAnd, ov::op::v1)
_OPENVINO_OP_REG(LogicalNot, ov::op::v1)
_OPENVINO_OP_REG(LogicalOr, ov::op::v1)
_OPENVINO_OP_REG(LogicalXor, ov::op::v1)
_OPENVINO_OP_REG(LRN, ov::op::v0)
_OPENVINO_OP_REG(LSTMCell, ov::op::v0)
_OPENVINO_OP_REG(MatMul, ov::op::v0)
_OPENVINO_OP_REG(MaxPool, ov::op::v1)
_OPENVINO_OP_REG(Maximum, ov::op::v1)
_OPENVINO_OP_REG(Minimum, ov::op::v1)
_OPENVINO_OP_REG(Mod, ov::op::v1)
_OPENVINO_OP_REG(Multiply, ov::op::v1)

_OPENVINO_OP_REG(MVN, ov::op::v0)  // Missing in opset1

_OPENVINO_OP_REG(Negative, ov::op::v0)
_OPENVINO_OP_REG(NonMaxSuppression, ov::op::v1)
_OPENVINO_OP_REG(NormalizeL2, ov::op::v0)
_OPENVINO_OP_REG(NotEqual, ov::op::v1)
_OPENVINO_OP_REG(OneHot, ov::op::v1)
_OPENVINO_OP_REG(PRelu, ov::op::v0)
_OPENVINO_OP_REG(PSROIPooling, ov::op::v0)
_OPENVINO_OP_REG(Pad, ov::op::v1)
_OPENVINO_OP_REG(Parameter, ov::op::v0)
_OPENVINO_OP_REG(Power, ov::op::v1)
_OPENVINO_OP_REG(PriorBox, ov::op::v0)
_OPENVINO_OP_REG(PriorBoxClustered, ov::op::v0)
_OPENVINO_OP_REG(Proposal, ov::op::v0)
_OPENVINO_OP_REG(Range, ov::op::v0)
_OPENVINO_OP_REG(Relu, ov::op::v0)
_OPENVINO_OP_REG(ReduceMax, ov::op::v1)
_OPENVINO_OP_REG(ReduceLogicalAnd, ov::op::v1)
_OPENVINO_OP_REG(ReduceLogicalOr, ov::op::v1)
_OPENVINO_OP_REG(ReduceMean, ov::op::v1)
_OPENVINO_OP_REG(ReduceMin, ov::op::v1)
_OPENVINO_OP_REG(ReduceProd, ov::op::v1)
_OPENVINO_OP_REG(ReduceSum, ov::op::v1)
_OPENVINO_OP_REG(RegionYolo, ov::op::v0)

_OPENVINO_OP_REG(ReorgYolo, ov::op::v0)  // Missing in opset1

_OPENVINO_OP_REG(Reshape, ov::op::v1)
_OPENVINO_OP_REG(Result, ov::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// _OPENVINO_OP_REG(Reverse, ov::op::v1)

_OPENVINO_OP_REG(ReverseSequence, ov::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// _OPENVINO_OP_REG(RNNCell, ov::op::v0)

_OPENVINO_OP_REG(ROIPooling, ov::op::v0)  // Missing in opset1

_OPENVINO_OP_REG(Select, ov::op::v1)
_OPENVINO_OP_REG(Selu, ov::op::v0)
_OPENVINO_OP_REG(ShapeOf, ov::op::v0)

// Moved out of opset2, it was added to opset1 by mistake
// _OPENVINO_OP_REG(ShuffleChannels, ov::op::v0)

_OPENVINO_OP_REG(Sign, ov::op::v0)
_OPENVINO_OP_REG(Sigmoid, ov::op::v0)
_OPENVINO_OP_REG(Sin, ov::op::v0)
_OPENVINO_OP_REG(Sinh, ov::op::v0)
_OPENVINO_OP_REG(Softmax, ov::op::v1)
_OPENVINO_OP_REG(Sqrt, ov::op::v0)
_OPENVINO_OP_REG(SpaceToDepth, ov::op::v0)
_OPENVINO_OP_REG(Split, ov::op::v1)
_OPENVINO_OP_REG(SquaredDifference, ov::op::v0)
_OPENVINO_OP_REG(Squeeze, ov::op::v0)
_OPENVINO_OP_REG(StridedSlice, ov::op::v1)
_OPENVINO_OP_REG(Subtract, ov::op::v1)
_OPENVINO_OP_REG(Tan, ov::op::v0)
_OPENVINO_OP_REG(Tanh, ov::op::v0)
_OPENVINO_OP_REG(TensorIterator, ov::op::v0)
_OPENVINO_OP_REG(Tile, ov::op::v0)
_OPENVINO_OP_REG(TopK, ov::op::v1)
_OPENVINO_OP_REG(Transpose, ov::op::v1)
_OPENVINO_OP_REG(Unsqueeze, ov::op::v0)
_OPENVINO_OP_REG(VariadicSplit, ov::op::v1)

// Moved out of opset2, it was added to opset1 by mistake
// _OPENVINO_OP_REG(Xor, ov::op::v0)

// New operations added in opset2
_OPENVINO_OP_REG(Gelu, ov::op::v0)
_OPENVINO_OP_REG(BatchToSpace, ov::op::v1)
_OPENVINO_OP_REG(SpaceToBatch, ov::op::v1)
