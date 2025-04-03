# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.conv_dilations_rewriter - Rewrites the patten used to represent dilations
pat = SpaceToBatchND->DepthwiseConv2dNative->BatchToSpaceND
"""

import numpy as np
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable


def rewrite_conv_dilations(g, ops):

    pattern1 = \
        OpTypePattern("BatchToSpaceND", name="batch_to_space", inputs=[
            OpTypePattern("DepthwiseConv2dNative|Conv2D|Conv3D", name="conv", inputs=[
                OpTypePattern("SpaceToBatchND", name="space_to_batch", inputs=[
                    OpTypePattern("*"),
                    OpTypePattern("Const|ConstV2"),
                    OpTypePattern("*"),
                ]),
                OpTypePattern("*"),
            ]),
            OpTypePattern("Const|ConstV2"),
            OpTypePattern("*"),
        ])
    pattern2 = \
        OpTypePattern("BatchToSpaceND", name="batch_to_space", inputs=[
            OpTypePattern("Squeeze", name="squeeze", inputs=[
                OpTypePattern("Conv2D", name="conv", inputs=[
                    OpTypePattern("ExpandDims", name="expand", inputs=[
                        OpTypePattern("SpaceToBatchND", name="space_to_batch", inputs=[
                            OpTypePattern("*"),
                            OpTypePattern("Const|ConstV2"),
                            OpTypePattern("*"),
                        ]),
                        OpTypePattern("Const|ConstV2"),
                    ]),
                    OpTypePattern("*"),
                ]),
            ]),
            OpTypePattern("Const|ConstV2"),
            OpTypePattern("*"),
        ])

    for pattern in [pattern1, pattern2]:
        matcher = GraphMatcher(pattern, allow_reorder=False)
        match_results = list(matcher.match_ops(ops))
        for match_result in match_results:
            is_conv_1d = pattern is pattern2
            space_to_batch = match_result.get_op("space_to_batch")
            conv = match_result.get_op("conv")
            batch_to_space = match_result.get_op("batch_to_space")
            if is_conv_1d:
                expand = match_result.get_op("expand")
                expand_axis = expand.inputs[1].get_tensor_value(as_list=True)
                squeeze = match_result.get_op("squeeze")
                squeeze_axes = squeeze.get_attr_value("squeeze_dims")
                if expand_axis not in [1, -3] or squeeze_axes not in [[1], [-3]]:
                    continue

            block_shape1 = space_to_batch.inputs[1].get_tensor_value(as_list=True)
            block_shape2 = batch_to_space.inputs[1].get_tensor_value(as_list=True)

            if block_shape1 != block_shape2:
                continue
            ndims = 2 if is_conv_1d else len(block_shape1)

            if conv.get_attr_value("padding") != b"VALID":
                continue

            if space_to_batch.inputs[2].is_const() and batch_to_space.inputs[2].is_const():
                paddings = space_to_batch.inputs[2].get_tensor_value(as_list=True)
                crops = batch_to_space.inputs[2].get_tensor_value(as_list=True)
                base_start_pad = [p[0] for p in paddings]
                if any(c[0] != 0 for c in crops):
                    continue
                base_end_pad = [p[1] - c[1] for p, c in zip(paddings, crops)]
                if not all(0 <= p[1] - bp < bs for p, bp, bs in zip(paddings, base_end_pad, block_shape1)):
                    continue
                pad_mode = "EXPLICIT"
                if is_conv_1d:
                    base_start_pad = [0] + base_start_pad
                    base_end_pad = [0] + base_end_pad
                base_pad_flat = [0, 0] + [x for s, e in zip(base_start_pad, base_end_pad) for x in [s, e]] + [0, 0]
            else:
                pad_mode = determine_pad_mode(space_to_batch.inputs[2])
                if pad_mode is None:
                    continue

            if is_conv_1d:
                block_shape1 = [1] + block_shape1
                inp = space_to_batch.input[0]
                g.replace_inputs(expand, [inp, expand.input[1]])
                g.copy_shape(batch_to_space.output[0], squeeze.output[0])
                g.replace_all_inputs(batch_to_space.output[0], squeeze.output[0])
                squeeze_out_shape = g.get_shape(squeeze.output[0])
                g.set_shape(squeeze.input[0], squeeze_out_shape[:1] + [1] + squeeze_out_shape[1:])
                expand_inp_shape = g.get_shape(expand.input[0])
                g.set_shape(expand.output[0], expand_inp_shape[:1] + [1] + expand_inp_shape[1:])
            else:
                inp = space_to_batch.input[0]
                kernel = conv.input[1]
                g.replace_inputs(conv, [inp, kernel])
                g.copy_shape(batch_to_space.output[0], conv.output[0])
                g.replace_all_inputs(batch_to_space.output[0], conv.output[0])

            if conv.get_attr_value("data_format") in [b"NCHW", b"NCDHW"]:
                conv.set_attr("dilations", [1] + block_shape1)
            else:
                conv.set_attr("dilations", [1] + block_shape1 + [1])
            conv.set_attr("padding", pad_mode)
            if pad_mode == "EXPLICIT":
                conv.set_attr("explicit_paddings", base_pad_flat)

    return g.get_nodes()

def determine_pad_mode(paddings_inp_node):
    tensor_ops = set(["Concat", "ConcatV2", "ConcatV3", "StridedSlice", "Pack", "ExpandDims", "Identity"])
    while paddings_inp_node.type in tensor_ops:
        non_const = [inp for inp in paddings_inp_node.inputs if not inp.is_const()]
        if len(non_const) == 0:
            return None
        paddings_inp_node = non_const[0]
    if paddings_inp_node.type == "FloorMod":
        return "VALID"
    if paddings_inp_node.type in ["Add", "AddV2"]:
        if paddings_inp_node.inputs[0].type == "FloorMod":
            pad = paddings_inp_node.inputs[1]
        elif paddings_inp_node.inputs[1].type == "FloorMod":
            pad = paddings_inp_node.inputs[0]
        else:
            return None
        if pad.is_const():
            if np.any(pad.get_tensor_value(as_list=False)):
                #return "SAME"   ORT doesn't implement dilations for SAME autopadding yet
                return None
            return "VALID"
    return None
