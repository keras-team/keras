# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx conv2d op with BiasAdd
"""
from tf2onnx import logging
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring

def rewrite_biasadd_with_conv2d(g, ops):
    pattern1 = \
        OpTypePattern('BiasAdd', name='biasadd', inputs=[
            OpTypePattern('Conv2D|Conv2DBackpropInput', name='conv', inputs=['*', '*']), '*'])
    pattern2 = \
        OpTypePattern('BiasAdd', name='biasadd', inputs=[
            OpTypePattern('Conv2D|Conv2DBackpropInput', name='conv', inputs=[
                '*', '*', '*']), '*'], allow_reorder=True)

    for pattern in [pattern1, pattern2]:
        matcher = GraphMatcher(pattern)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            biasadd = match.get_op('biasadd')
            conv = match.get_op('conv')

            # Backup the conv and biasadd values
            conv_type = conv.type
            conv_input = conv.input
            conv_attr = conv.attr
            dtype = g.get_dtype(conv.output[0])
            shape = g.get_shape(conv.output[0])
            conv_name = biasadd.name
            conv_output = biasadd.output
            if pattern == pattern2:
                conv_inputs = [conv_input[0], conv_input[1], conv_input[2], biasadd.input[1]]
            else:
                conv_inputs = [conv_input[0], conv_input[1], biasadd.input[1]]

            if len(g.find_output_consumers(conv.output[0])) > 1:
                continue
            # Remove the Conv and BiasAdd node
            g.remove_node(conv.name)
            g.remove_node(biasadd.name)

            g.make_node(conv_type, conv_inputs, attr=conv_attr, name=conv_name, outputs=conv_output,
                        shapes=[shape], dtypes=[dtype], skip_conversion=False)
    return ops
