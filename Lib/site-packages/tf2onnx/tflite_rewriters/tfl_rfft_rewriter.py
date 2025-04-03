# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.tflite_rewriters.tfl_rfft_zero_rewriter - TFLite uses RFFT2D -> Reshape sequence to do 1D RFFT
"""

import numpy as np

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx import utils


# pylint: disable=missing-docstring,unused-argument

def rewrite_tfl_rfft(g, ops):
    pattern0 = \
        OpTypePattern('TFL_COMPLEX_ABS', name='complex_abs', inputs=[
            OpTypePattern('TFL_RESHAPE', name='reshape', inputs=[
                OpTypePattern('TFL_RFFT2D', name='rfft2d', inputs=[
                    OpTypePattern('*'),
                    OpTypePattern('Const|ConstV2', name='length'),
                ]),
                OpTypePattern('Const|ConstV2', name='shape'),
            ], allow_reorder=True),
        ])

    matcher = GraphMatcher(pattern0, allow_reorder=False)
    match_results = list(matcher.match_ops(ops))
    if match_results:
        for match in match_results:
            length = match.get_op("length").get_tensor_value(as_list=True)
            rfft2d = match.get_op("rfft2d")
            complex_abs = match.get_op("complex_abs")
            reshape = match.get_op("reshape")
            shape = match.get_op("shape").get_tensor_value(as_list=True)
            output_shape = g.get_shape(rfft2d.output[0])

            if output_shape is None or output_shape != shape[:-1] + [1, shape[-1]]:
                continue
            if length[0] != 1:
                continue

            rfft2d.type = "RFFT"
            g.copy_shape(complex_abs.input[0], rfft2d.output[0])
            # Skip the Reshape
            g.replace_input(complex_abs, complex_abs.input[0], rfft2d.output[0], 0)

            new_length = g.make_const(utils.make_name("rfft_length"), np.array([length[1]], np.int64))
            g.replace_input(rfft2d, rfft2d.input[1], new_length.output[0], 1)

            g.replace_all_inputs(complex_abs.output[0], reshape.output[0])
            # Move reshape below complex abs
            g.replace_input(reshape, reshape.input[0], complex_abs.output[0], 0)

    return ops
