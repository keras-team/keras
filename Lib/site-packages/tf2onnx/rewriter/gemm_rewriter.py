# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewrite - rewrite tensorflow subgraph to onnx gemm op
"""
import logging
from onnx import onnx_pb
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring

def rewrite_gemm(g, ops):
    if g.opset <= 6:
        return ops

    # pattern0: alpha*A*B + beta*C
    pattern0 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('Const', name='alpha'),
                OpTypePattern('MatMul', name='matmul')
            ]),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C')
            ])
        ])

    # pattern1: alpha*A*B + C
    pattern1 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('Mul', name='mul1', inputs=[
                OpTypePattern('MatMul', name='matmul'),
                OpTypePattern('Const', name='alpha')
            ]),
            OpTypePattern('*', name='C'),
        ])

    # pattern2: A*B + beta*C
    pattern2 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('Mul', name='mul2', inputs=[
                OpTypePattern('Const', name='beta'),
                OpTypePattern('*', name='C')
            ])
        ])

    # pattern3: A*B + C
    pattern3 = \
        OpTypePattern('Add|AddV2', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('*', name='C'),
        ])

    # pattern4: A*B + c
    pattern4 = \
        OpTypePattern('BiasAdd', name='add', inputs=[
            OpTypePattern('MatMul', name='matmul'),
            OpTypePattern('*', name='C'),
        ])

    pattern_list = [pattern0, pattern1, pattern2, pattern3, pattern4]

    for pattern in pattern_list:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        if match_results:
            for match in match_results:
                matmul_node = match.get_op("matmul")

                if g.get_dtype(matmul_node.input[0]) != onnx_pb.TensorProto.FLOAT:
                    logging.warning(u"For now, onnxruntime only support float32 type for Gemm rewriter")
                    continue

                attr, is_valid = get_gemm_attr(match)
                if not is_valid:
                    continue

                add_node = match.get_op('add')
                a_edge_name = matmul_node.input[0]
                b_edge_name = matmul_node.input[1]
                c_edge_name = match.get_tensor("C")

                a_mul_b_shape = g.get_shape(matmul_node.output[0])
                c_shape = g.get_shape(c_edge_name)
                if c_shape is None: continue
                if a_mul_b_shape is None: continue
                if -1 in c_shape + a_mul_b_shape: continue
                if g.get_rank(a_edge_name) != 2 or g.get_rank(b_edge_name) != 2: continue
                compatible = True
                for i in range(1, len(c_shape) + 1):
                    if c_shape[-i] not in [1, a_mul_b_shape[-i]]:
                        compatible = False
                if not compatible: continue

                gemm = g.make_node("Gemm", inputs=[a_edge_name, b_edge_name, c_edge_name],
                                   attr=attr,
                                   shapes=[g.get_shape(add_node.output[0])],
                                   dtypes=[g.get_dtype(add_node.output[0])], op_name_scope=matmul_node.name)

                ops.append(gemm)
                g.replace_all_inputs(add_node.output[0], gemm.output[0], ops=ops)
                to_delete = [add_node, matmul_node]
                g.safe_remove_nodes(to_delete)
    return ops


def get_gemm_attr(match):
    attr = {}
    for arg in ["alpha", "beta"]:
        arg_op = match.get_op(arg)
        if arg_op is not None:
            match_args = arg_op.get_tensor_value()
            if isinstance(match_args, list):
                if len(match_args) != 1:
                    return attr, False
                match_args = match_args[0]
            attr[arg] = match_args
    for arg in ["matmul"]:
        arg_op = match.get_op(arg)
        if arg_op is not None:
            match_args = arg_op.attr
            if isinstance(match_args, dict):
                keys = list(match_args.keys())
                if 'transpose_a' not in keys and 'transpose_b' not in keys:
                    return attr, False
                match_args_a = match_args['transpose_a'].i
                attr['transA'] = match_args_a
                match_args_b = match_args['transpose_b'].i
                attr['transB'] = match_args_b
    return attr, True
