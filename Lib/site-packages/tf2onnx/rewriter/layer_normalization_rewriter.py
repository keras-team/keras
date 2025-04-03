# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewrite - Rewrites a pattern from the tf layer_norm contrib op.
Converts a mean/variance normalization pattern (using ReduceMean, RSqrt, Sub, Mul, etc.) into InstanceNormalization
"""
from onnx import TensorProto, helper
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.graph_builder import GraphBuilder


# pylint: disable=missing-docstring

def rewrite_layer_normalization(g, ops):
    # Needs ConstantOfShape
    if g.opset <= 9:
        return ops

    inner_pattern = \
        OpTypePattern('Rsqrt', inputs=[
            OpTypePattern('Add', inputs=[
                OpTypePattern('Mean', allow_reorder=False, inputs=[
                    OpTypePattern('Square', inputs=[
                        OpTypePattern('Sub', allow_reorder=False, inputs=[
                            OpTypePattern('*', name='input'),
                            OpTypePattern('Mean', name='mean', allow_reorder=False, inputs=[
                                OpTypePattern('*', name='input_r2'),
                                OpTypePattern('Const|ConstV2', name='mean_axes')
                            ])
                        ])
                    ]),
                    OpTypePattern('Const|ConstV2', name='variance_axes')
                ]),
                OpTypePattern('Const|ConstV2', name='epsilon')
            ])
        ])

    pattern0 = \
        OpTypePattern('Add', name='bias_add', inputs=[
            OpTypePattern('Mul', name='scale_mul', inputs=[
                OpTypePattern('Mul', inputs=[
                    inner_pattern,
                    OpTypePattern('*', name='scale')
                ]),
                OpTypePattern('Sub', inputs=[
                    OpTypePattern('*', name='input_r3'),
                    OpTypePattern('Mean', name='mean_r2')
                ])
            ]),
            OpTypePattern('*', name='bias')
        ])
    pattern1 = \
        OpTypePattern('Add', name='bias_add', inputs=[
            OpTypePattern('Mul', name='scale_mul', inputs=[
                OpTypePattern('Mul', inputs=[
                    inner_pattern,
                    OpTypePattern('Sub', inputs=[
                        OpTypePattern('*', name='input_r3'),
                        OpTypePattern('Mean', name='mean_r2')
                    ])
                ]),
                OpTypePattern('*', name='scale')
            ]),
            OpTypePattern('*', name='bias'),
        ])
    pattern2 = \
        OpTypePattern('Add', name='bias_add', inputs=[
            OpTypePattern('Mul', name='scale_mul', inputs=[
                OpTypePattern('Mul', inputs=[
                    OpTypePattern('*', name='scale'),
                    OpTypePattern('Sub', inputs=[
                        OpTypePattern('*', name='input_r3'),
                        OpTypePattern('Mean', name='mean_r2')
                    ])
                ]),
                inner_pattern
            ]),
            OpTypePattern('*', name='bias'),
        ])

    pattern_list = [pattern0, pattern1, pattern2]

    for pattern in pattern_list:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        if match_results:
            for match in match_results:
                input_tensor = match.get_tensor('input')
                rank = g.get_rank(input_tensor)
                node = match.get_op('bias_add')
                if input_tensor != match.get_tensor('input_r2') or input_tensor != match.get_tensor('input_r3'):
                    continue
                if match.get_op('mean').name != match.get_op('mean_r2').name:
                    continue
                inp = match.get_op('mean').input[0]
                if rank != 3:
                    continue
                mean_axes = match.get_op('mean_axes').get_tensor_value(as_list=True)
                variance_axes = match.get_op('variance_axes').get_tensor_value(as_list=True)
                mean_axes = [a % rank for a in mean_axes]
                variance_axes = [a % rank for a in variance_axes]
                if mean_axes != [2] or variance_axes != [2]:
                    continue
                epsilon = match.get_op('epsilon').get_tensor_value(as_list=False).flatten().tolist()
                if len(epsilon) != 1:
                    continue
                scale = match.get_tensor('scale')
                bias = match.get_tensor('bias')
                shape = g.make_node("Shape", [inp]).output[0]
                dim_2_shape = GraphBuilder(g).make_slice(
                    {"data": shape, "ends": [2], "starts": [1], "axes": [0]})
                zero_tensor = helper.make_tensor("value", TensorProto.FLOAT, dims=[1], vals=[0])
                one_tensor = helper.make_tensor("value", TensorProto.FLOAT, dims=[1], vals=[1])
                zeros_of_shape = g.make_node("ConstantOfShape", [dim_2_shape], attr={'value': zero_tensor}).output[0]
                ones_of_shape = g.make_node("ConstantOfShape", [dim_2_shape], attr={'value': one_tensor}).output[0]
                norm = g.make_node("InstanceNormalization", [inp, ones_of_shape, zeros_of_shape],
                                   attr={'epsilon': epsilon[0]}, op_name_scope=node.name).output[0]
                mul = g.make_node("Mul", [norm, scale]).output[0]
                add = g.make_node("Add", [mul, bias]).output[0]
                g.replace_all_inputs(node.output[0], add)
                g.remove_node(node.name)
    return ops
