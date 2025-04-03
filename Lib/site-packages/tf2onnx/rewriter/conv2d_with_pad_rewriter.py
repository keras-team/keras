# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter - rewrite tensorflow subgraph to onnx condv2 op with pad
"""

import numpy as np

from tf2onnx import handler, logging
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring


def rewrite_conv2d_with_pad(g, ops):
    pattern = \
        OpTypePattern("Conv2D", name="conv", inputs=[
            OpTypePattern("Pad", name="pad"),
            OpTypePattern("*")
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    for match in match_results:
        conv = match.get_op("conv")
        pad = match.get_op("pad")
        paddings = pad.inputs[1]

        if not paddings.is_const():
            continue
        mode = pad.get_attr("mode")
        if mode:
            mode = mode.s.decode("utf-8").lower()
        if mode not in [None, "constant"] or len(pad.input) >= 3:
            continue
        # Conv2D already has a pad
        if conv.get_attr("padding").s.decode("utf-8") == "SAME":
            continue

        logger.debug("merge pad [%s] into conv [%s]", pad.name, conv.name)
        paddings_val = np.array(paddings.get_tensor_value())
        # can't pad on batch or channel dimensions
        data_format = conv.get_attr("data_format").s.decode("utf-8")
        if data_format == "NHWC":
            if np.any(paddings_val[0]) or np.any(paddings_val[3]):
                continue
            paddings_val = paddings_val[1:3]
        else:
            if np.any(paddings_val[0]) or np.any(paddings_val[1]):
                continue
            paddings_val = paddings_val[2:4]

        paddings_val = paddings_val.transpose().flatten()
        g.replace_input(conv, conv.input[0], pad.input[0], 0)
        # convert Conv2D
        conv.type = "Conv2D"
        func, _ = handler.tf_op.find_effective_op("Conv2D")
        func(g, conv)
        conv.skip_conversion = True
        conv.set_attr("auto_pad", "NOTSET")
        conv.set_attr("pads", paddings_val)
    return ops
