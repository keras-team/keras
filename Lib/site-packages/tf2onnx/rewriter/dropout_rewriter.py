# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx dropout op
"""

import numpy as np
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx import logging

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring


def rewrite_dropout(g, ops):
    patterns = [
        OpTypePattern('Mul', name='outputs', inputs=[
            OpTypePattern('RealDiv', name="input2"),
            OpTypePattern('Floor', inputs=[
                OpTypePattern('Add', inputs=[
                    OpTypePattern("*", name="input3"),
                    OpTypePattern('RandomUniform|RandomUniformLike'),
                ])
            ]),
        ]),
        OpTypePattern("Mul", name="outputs", inputs=[
            OpTypePattern("Mul", name="input2"),
            OpTypePattern("Cast", inputs=[
                OpTypePattern("GreaterEqual", inputs=[
                    OpTypePattern("RandomUniform|RandomUniformLike"),
                    OpTypePattern("*", name="input3")
                ])
            ])
        ]),
        # pattern for tf-2.0 tf.nn.dropout()
        OpTypePattern("Mul", name="outputs", inputs=[
            OpTypePattern("Cast", inputs=[
                OpTypePattern("GreaterEqual", inputs=[
                    OpTypePattern("RandomUniform|RandomUniformLike"),
                    OpTypePattern("*", name="input3")
                ])
            ]),
            OpTypePattern("Mul", name="input2"),
        ]),
    ]
    for pattern in patterns:
        matcher = GraphMatcher(pattern, allow_reorder=True)
        match_results = list(matcher.match_ops(ops))
        for match in match_results:
            input2 = match.get_op('input2')
            input3 = match.get_op('input3')
            outputs = match.get_op('outputs')

            if not input3.is_scalar():
                logger.warning("Dropout pattern rooted at %s does not have a "
                               "constant ratio and cannot be replaced.", outputs.name)
                continue
            ratio = input3.get_tensor_value()

            if input2.inputs[0].is_scalar():
                data_output = input2.input[1]
                scaling_constant = input2.inputs[0].get_tensor_value()
            elif input2.inputs[1].is_scalar():
                data_output = input2.input[0]
                scaling_constant = input2.inputs[1].get_tensor_value()
            else:
                logger.warning("Could not find scaling constant for dropout pattern rooted at %s. "
                               "The pattern will not be replaced with an ONNX dropout node.", outputs.name)
                continue

            #The scaling constant should be 1/(1-ratio), otherwise this isn't truly a dropout node
            if not np.allclose([1], [scaling_constant * (1 - ratio)]):
                logger.warning("Scaling constant %f for dropout pattern rooted at %s is inconsistent with dropout "
                               "ratio %f. The pattern will not be replaced with an ONNX dropout node.",
                               scaling_constant, outputs.name, ratio)
                continue

            nodes_to_remove = [n for n in match.get_nodes() if n.name != input3.name]
            if not g.is_safe_to_remove_nodes(nodes_to_remove, [outputs.output[0]]):
                logger.warning("Nodes in dropout pattern rooted at %s cannot be removed because intermediate results "
                               "of some nodes are referenced elsewhere in graph.", outputs.name)
                continue

            op_name = utils.make_name("Dropout")
            out_name = utils.port_name(op_name)
            new_node = g.make_node(
                "Dropout",
                inputs=[data_output],
                outputs=[out_name],
                name=op_name,
                attr={"ratio": ratio},
                shapes=[g.get_shape(data_output)],
                dtypes=[g.get_dtype(data_output)]
            )
            g.replace_all_inputs(outputs.output[0], new_node.output[0], ops=ops)
            for n in nodes_to_remove:
                g.remove_node(n.name)

    return ops
