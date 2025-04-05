/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GRAPH_INTERFACE_OPSET_HPP
#define GRAPH_INTERFACE_OPSET_HPP

#include <functional>

#include "graph/interface/op_def.hpp"
#include "graph/interface/op_schema.hpp"

namespace dnnl {
namespace impl {
namespace graph {

class opset_v1_t {
public:
    static void for_each_schema(const std::function<void(op_schema_t &&)> &fn) {
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Abs, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(AbsBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Add, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(AvgPool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        AvgPoolBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BatchNormInference, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BatchNormForwardTraining, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BatchNormTrainingBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(BiasAdd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        BiasAddBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Clamp, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ClampBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Concat, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Convolution, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvolutionBackwardData, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvolutionBackwardWeights, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ConvTranspose, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvTransposeBackwardData, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        ConvTransposeBackwardWeights, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Dequantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Divide, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Elu, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(EluBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(End, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Exp, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(GELUBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(HardSigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        HardSigmoidBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(HardSwish, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        HardSwishBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Interpolate, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        InterpolateBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LayerNorm, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        LayerNormBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LeakyReLU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Log, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(LogSoftmax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        LogSoftmaxBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MatMul, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Maximum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MaxPool, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        MaxPoolBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Minimum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Mish, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(MishBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Multiply, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Pow, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PReLU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(PReLUBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Quantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceL1, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceL2, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceMax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceMean, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceMin, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceProd, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReduceSum, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLU, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(ReLUBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Reorder, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Round, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Select, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Sigmoid, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SigmoidBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftMax, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SoftMaxBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SoftPlus, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SoftPlusBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Sqrt, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(SqrtBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Square, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        SquaredDifference, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(StaticReshape, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        StaticTranspose, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Subtract, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Tanh, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(TanhBackward, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Wildcard, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(TypeCast, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        DynamicQuantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(
                        DynamicDequantize, 1)>());
        fn(get_op_schema<DNNL_GRAPH_OP_SCHEMA_CLASS_NAME(Reciprocal, 1)>());
    }
};

inline void register_opset_schema() {
    register_opset_schema<opset_v1_t>();
}

} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
