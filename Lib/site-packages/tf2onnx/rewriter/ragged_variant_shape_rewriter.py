# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - RaggedTensorToVariant -> Shape pattern
"""

import numpy as np
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher


# pylint: disable=missing-docstring


def rewrite_ragged_variant_shape(g, ops):
    pattern1 = \
        OpTypePattern('Shape', name='shape', inputs=[
            OpTypePattern('RaggedTensorToVariant', name='raggedtovariant')
        ])

    pattern_list = [pattern1]
    for pattern in pattern_list:
        matcher = GraphMatcher(pattern)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            shape = match.get_op('shape')
            raggedtovariant = match.get_op('raggedtovariant')
            if raggedtovariant.get_attr_value("batched_input") != 1:
                continue
            if raggedtovariant.get_attr_value("RAGGED_RANK") != 1:
                continue
            # Shape of batched variant from ragged is same as number of splits minus 1
            g.replace_inputs(shape, [raggedtovariant.input[0]])
            np_dtype = utils.map_onnx_to_numpy_type(g.get_dtype(shape.output[0]))
            const_one = g.make_const(utils.make_name("const_one"), np.array(1, np_dtype)).output[0]
            g.insert_new_node_on_output("Sub", shape.output[0], inputs=[shape.output[0], const_one])

    return ops
