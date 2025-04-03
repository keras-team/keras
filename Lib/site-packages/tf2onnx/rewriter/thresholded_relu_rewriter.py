# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx ThresholdedRelu op
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_thresholded_relu(g, ops):
    if g.opset < 10:
        return ops

    pattern = \
        OpTypePattern('Mul', name='mul', inputs=[
            OpTypePattern('Cast', name='cast', inputs=[
                OpTypePattern('Greater', name='greater', inputs=[
                    OpTypePattern('*', name='greater_input'),
                    OpTypePattern('Const', name='theta')
                ])
            ]),
            OpTypePattern('*', name='mul_input')
        ])
    matcher = GraphMatcher(pattern, allow_reorder=True)
    match_results = list(matcher.match_ops(ops))

    for match in match_results:
        mul_node = match.get_op("mul")
        cast_node = match.get_op('cast')

        greater_input_edge_name = match.get_tensor('greater_input')
        mul_input_edge_name = match.get_tensor('mul_input')
        if greater_input_edge_name == mul_input_edge_name:
            theta = match.get_op('theta').get_tensor_value()
            thresholded_relu = g.make_node("ThresholdedRelu", inputs=[mul_input_edge_name], attr={"alpha": theta},
                                           shapes=[g.get_shape(mul_node.output[0])],
                                           dtypes=[g.get_dtype(mul_node.output[0])])
            g.replace_all_inputs(mul_node.output[0], thresholded_relu.output[0], ops=ops)
            to_delete = [cast_node, mul_node]
            g.safe_remove_nodes(to_delete)
    return ops
