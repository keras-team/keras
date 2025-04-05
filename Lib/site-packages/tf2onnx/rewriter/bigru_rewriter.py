# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.bigru_rewriter - bigru support.
This rewriter depends on tf2onnx.rewriter.gru_rewriter's results.
"""

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.rewriter import rnn_utils


logger = logging.getLogger(__name__)

# pylint: disable=invalid-name,unused-argument,missing-docstring

def process_bigru(g, bi_grus):
    for gru_fw, gru_bw in bi_grus:
        logger.debug("=========================")
        logger.debug("start handling potential bidirectional gru: %s, %s", gru_fw.name, gru_bw.name)

        w_fw = rnn_utils.get_np_val_for_const(g, gru_fw, 1)
        w_bw = rnn_utils.get_np_val_for_const(g, gru_bw, 1)
        r_fw = rnn_utils.get_np_val_for_const(g, gru_fw, 2)
        r_bw = rnn_utils.get_np_val_for_const(g, gru_bw, 2)
        b_fw = rnn_utils.get_np_val_for_const(g, gru_fw, 3)
        b_bw = rnn_utils.get_np_val_for_const(g, gru_bw, 3)
        W = np.concatenate((w_fw, w_bw), axis=0)
        R = np.concatenate((r_fw, r_bw), axis=0)
        B = np.concatenate((b_fw, b_bw), axis=0)

        all_nodes = g.get_nodes()
        if len(gru_fw.inputs) == len(gru_bw.inputs):
            if len(gru_fw.inputs) > 4:
                initializer_node = process_init_nodes(g, gru_fw, gru_bw, all_nodes)
        else:
            logger.error("fw, bw gru inputs num is not consistent. stop")
            continue

        # create node
        w_name = utils.make_name("W")
        w_node = g.make_const(w_name, W, skip_conversion=True)
        all_nodes.append(w_node)

        r_name = utils.make_name("R")
        r_node = g.make_const(r_name, R, skip_conversion=True)
        all_nodes.append(r_node)

        b_name = utils.make_name("B")
        b_node = g.make_const(b_name, B, skip_conversion=True)
        all_nodes.append(b_node)
        gru_inputs = [gru_fw.input[0], w_node.output[0],
                      r_node.output[0], b_node.output[0]]
        if len(gru_fw.inputs) > 4:
            gru_inputs.extend([gru_fw.input[4], initializer_node.output[0]])

        direction = "bidirectional"
        attr = {}
        for name in rnn_utils.onnx_rnn_attr_mapping[rnn_utils.ONNX_RNN_TYPE.GRU]:
            attr_val = gru_fw.get_attr_value(name)
            if attr_val:
                attr[name] = attr_val
        # activation has to be took care, attr here is proto
        activations = [act.decode("utf-8")
                       for act in gru_fw.get_attr_value("activations")]
        activations += [act.decode("utf-8")
                        for act in gru_bw.get_attr_value("activations")]
        attr.update({"direction": direction, "activations": activations})

        bi_gru_node = g.make_node("GRU", gru_inputs, attr=attr, output_count=2)
        all_nodes.append(bi_gru_node)
        logger.debug("processing output nodes")

        to_remove = [gru_fw.name, gru_fw.input[1], gru_fw.input[2], gru_fw.input[3],
                     gru_bw.name, gru_bw.input[1], gru_bw.input[2], gru_bw.input[3]]
        rnn_utils.slice_birnn_for_original_rnn_consumers(
            g, gru_fw, gru_bw, bi_gru_node, 0, all_nodes, to_remove)
        rnn_utils.slice_birnn_for_original_rnn_consumers(
            g, gru_fw, gru_bw, bi_gru_node, 1, all_nodes, to_remove)

        gru_bw_old_x = gru_bw.input[0]
        for n in to_remove:
            g.remove_node(n)

        rnn_utils.remove_reverse_in_bw_input(g, gru_bw_old_x, rnn_utils.ONNX_RNN_TYPE.GRU)

    return g.get_nodes()


def process_init_nodes(g, gru_fw, gru_bw, to_append):
    initializer_node = rnn_utils.process_single_init_node(
        g, gru_fw.input[5], gru_bw.input[5], to_append)

    return initializer_node


def rewrite_bidirectional_grus(g, ops):
    bi_grus = rnn_utils.find_bidirectional_rnns(g, ops, rnn_utils.ONNX_RNN_TYPE.GRU)

    return process_bigru(g, bi_grus)
