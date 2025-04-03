# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.late_rewriters.channel_order_rewriters - contains rewriters for replacing ops with channel first/last versions
"""

from tf2onnx import utils, constants

# pylint: disable=invalid-name,unused-argument,missing-docstring, unused-variable

_CHANNELS_FIRST_OPS = [
    "AveragePool",
    "BatchNormalization",
    "Conv",
    "ConvInteger",
    "ConvTranspose",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "InstanceNormalization",
    "LpPool",
    "LRN",
    "MaxPool",
    "MaxRoiPool",
    "MaxUnpool",
    "QLinearConv",
]


def channel_last_to_first_perm(rank):
    return [0, rank - 1] + list(range(1, rank - 1))


def channel_first_to_last_perm(rank):
    return [0] + list(range(2, rank)) + [1]


def _to_channel_last_handler(g, op):
    # For now, all ops can use the same handlers (input[0] and output[0] are always correct)
    rank = g.get_rank(op.output[0])
    utils.make_sure(rank is not None, "Cannot convert %s node %s with unknown rank to channels last", op.type, op.name)
    op.type = "ChannelsLast" + op.type
    op.domain = constants.CONTRIB_OPS_DOMAIN
    inp_perm = channel_first_to_last_perm(rank)
    out_perm = channel_last_to_first_perm(rank)
    output_shape = g.get_shape(op.output[0])
    if output_shape is not None:
        output_shape = [output_shape[i] for i in inp_perm]
    g.set_shape(op.output[0], output_shape)

    g.insert_new_node_on_input(op, "Transpose", op.input[0], input_index=0, perm=inp_perm)
    g.insert_new_node_on_output("Transpose", op.output[0], perm=out_perm)


def _to_channel_first_handler(g, op):
    rank = g.get_rank(op.output[0])
    utils.make_sure(rank is not None, "Cannot convert %s node %s with unknown rank to channels last", op.type, op.name)
    op.type = op.type.replace("ChannelsLast", "")
    op.domain = constants.ONNX_DOMAIN
    inp_perm = channel_last_to_first_perm(rank)
    out_perm = channel_first_to_last_perm(rank)
    output_shape = g.get_shape(op.output[0])
    if output_shape is not None:
        output_shape = [output_shape[i] for i in inp_perm]
    g.set_shape(op.output[0], output_shape)

    g.insert_new_node_on_input(op, "Transpose", op.input[0], input_index=0, perm=inp_perm)
    g.insert_new_node_on_output("Transpose", op.output[0], perm=out_perm)


def get_channels_first_ops(opset=None):
    # opset doesn't matter for now
    return set(_CHANNELS_FIRST_OPS)


def rewrite_channels_last(g, ops):
    channel_first_ops = get_channels_first_ops(g.opset)
    for op in ops:
        if op.type in channel_first_ops:
            _to_channel_last_handler(g, op)
    return g.get_nodes()


def rewrite_channels_first(g, ops):
    for op in ops:
        if op.type.startswith("ChannelsLast"):
            _to_channel_first_handler(g, op)
    return g.get_nodes()
