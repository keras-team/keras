# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx leakyrelu op
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_leakyrelu(g, ops):
    if g.opset < 6:
        return ops

    pattern = \
        OpTypePattern('Maximum', name='max', inputs=[
            OpTypePattern('Mul', name='mul', inputs=[
                OpTypePattern('Const', name='alpha'),
                OpTypePattern('*', name='mul_input'),
            ]),
            OpTypePattern('*', name='max_input'),
        ])

    matcher = GraphMatcher(pattern, allow_reorder=True)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        max_node = match.get_op('max')
        mul_node = match.get_op("mul")

        max_input_edge_name = match.get_tensor('max_input')
        mul_input_edge_name = match.get_tensor('mul_input')
        if max_input_edge_name == mul_input_edge_name:
            alpha = match.get_op("alpha").get_tensor_value()
            if alpha >= 1:
                continue
            leakyrelu = g.make_node("LeakyRelu", inputs=[max_input_edge_name], attr={"alpha": alpha},
                                    shapes=[g.get_shape(max_node.output[0])], dtypes=[g.get_dtype(max_node.output[0])])
            ops.append(leakyrelu)
            g.replace_all_inputs(max_node.output[0], leakyrelu.output[0], ops=ops)
            to_delete = [max_node, mul_node]
            g.safe_remove_nodes(to_delete)

    return ops
