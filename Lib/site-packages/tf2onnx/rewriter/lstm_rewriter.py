# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.lstm_rewriter
"""

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.rewriter.rnn_utils import RNNUnitType, get_weights_from_const_node
from tf2onnx.utils import is_tf_concat_op, is_tf_slice_op

from tf2onnx.rewriter.lstm_rewriter_base import LSTMRewriterBase

# pylint: disable=invalid-name,unused-argument,missing-docstring


logger = logging.getLogger(__name__)


class LSTMRewriter(LSTMRewriterBase):
    def __init__(self, g):
        super(LSTMRewriter, self).__init__(g)
        self.lstm_cell_type = None
        self.num_lstm_layers = 0

    def run(self):
        logger.debug("enter lstm rewriter")
        return super(LSTMRewriter, self).run()

    def find_cell(self, context):
        lstm_cell_types = [RNNUnitType.LSTMCell, RNNUnitType.LSTMBlockCell]
        for cell_type in lstm_cell_types:
            cell_match = self._match_cell(context, cell_type)
            if cell_match and len(cell_match) >= 1:
                self.num_lstm_layers = len(cell_match)
                logger.debug("number of LSTM layers: %s", self.num_lstm_layers)
                for i in range(self.num_lstm_layers):
                    self.state_variable_handlers.append({
                        "ct" + str(i): (self._ct_variable_finder, self._connect_lstm_yc_to_graph, i),
                        "ht" + str(i): (self._ht_variable_finder, self._connect_lstm_yh_to_graph, i)
                    })
                    self.state_variable_handlers.append({
                        "ct_ht" + str(i): (self._ct_ht_shared_variable_finder, self._connect_lstm_ych_to_graph, i)
                    })
                logger.debug("parsing unit is %s, num layers is %d", cell_type, self.num_lstm_layers)
            if cell_match:
                self.lstm_cell_type = cell_type
                logger.debug("parsing unit is %s", cell_type)
                return cell_match
        logger.debug("cannot parse unit")
        return None

    def get_weight_and_bias(self, context):
        weight_and_bias = list()
        for i in range(self.num_lstm_layers):
            if self.lstm_cell_type == RNNUnitType.LSTMCell:
                weight_and_bias.append(self._get_weight_and_bias_for_lstm_cell(context, i))
            if self.lstm_cell_type == RNNUnitType.LSTMBlockCell:
                weight_and_bias.append(self._get_weight_and_bias_for_lstmblock_cell(context, i))
        return weight_and_bias

    def _get_weight_and_bias_for_lstmblock_cell(self, context, i):
        cell_match = context.cell_match[i]

        w_node = cell_match.get_op("cell_kernel")
        w = get_weights_from_const_node(self.g, w_node)
        if w is None:
            logger.warning("Cannot find weight, SKIP")
            return None

        b_node = cell_match.get_op("cell_bias")
        b = get_weights_from_const_node(self.g, b_node)
        if b is None or b.shape[0] != w.shape[1]:
            logger.warning("cell_kernel and cell_bias's dimension doesn't match, SKIP")
            return None

        lstm_block_cell = cell_match.get_op("lstm_block_cell")
        ft_bias_val = np.array(
            lstm_block_cell.get_attr("forget_bias").f,
            dtype=b.dtype
        )

        return {
            "weight": w,
            "bias": b,
            "ft_bias": ft_bias_val
        }

    def _get_weight_and_bias_for_lstm_cell(self, context, i):
        match = context.cell_match[i]

        w_e = match.get_op("cell_kernel")
        w = get_weights_from_const_node(self.g, w_e)
        if w is None or w.size == 0:
            return None

        # check https://www.tensorflow.org/versions/r1.8/api_docs/cc/class/tensorflow/ops/bias-add
        # for bias_add data format
        bias_add = match.get_op("bias_add")
        if bias_add is not None and bias_add.data_format != "NHWC":
            logger.debug("BiasAdd data_format is not NHWC, SKIP")
            return None

        b_e = match.get_op("cell_bias")
        if b_e is None:
            b = np.array([0 for i in range(len(w[0]))]).astype(w.dtype)
        else:
            b = get_weights_from_const_node(self.g, b_e)
            if b is None or b.shape[0] != w.shape[1]:
                logger.warning("cell_kernel and cell_bias's dimensions does not match, skip")
                return None

        ft_bias_node = match.get_op("ft_bias")
        ft_bias = get_weights_from_const_node(self.g, ft_bias_node)
        if ft_bias is None:
            return None

        if not b.dtype == ft_bias.dtype:
            return None

        return {
            "weight": w,
            "bias": b,
            "ft_bias": ft_bias
        }

    def parse_attributes(self, context):
        if self.lstm_cell_type == RNNUnitType.LSTMBlockCell:
            lstm_block_cell = context.cell_match[0].get_op("lstm_block_cell")
            clip = float(lstm_block_cell.get_attr("cell_clip").f)
            # current LSTM op cannot handle clip
            if clip > 0:
                return False

            use_peephole = lstm_block_cell.get_attr_value("use_peephole")
            if use_peephole:
                return False
        return True

    def _ct_variable_finder(self, context, i):
        if self.lstm_cell_type == RNNUnitType.LSTMCell:
            lstm_cell = context.cell_match[i]
            return self._find_state_variable_with_select(
                context,
                lstm_cell.get_op("ct").output[0],
                [lstm_cell.get_op("ct_identity_consumer")]
            )
        if self.lstm_cell_type == RNNUnitType.LSTMBlockCell:
            lstm_block_cell = context.cell_match[i].get_op("lstm_block_cell")
            return self._find_state_variable_with_select(
                context,
                lstm_block_cell.output[1],
                [lstm_block_cell]
            )
        return None

    def _ht_variable_finder(self, context, i):
        if self.lstm_cell_type == RNNUnitType.LSTMCell:
            lstm_cell = context.cell_match[i]
            return self._find_state_variable_with_select(
                context,
                lstm_cell.get_op("ht").output[0],
                [lstm_cell.get_op("xh")]
            )
        if self.lstm_cell_type == RNNUnitType.LSTMBlockCell:
            lstm_block_cell = context.cell_match[i].get_op("lstm_block_cell")
            return self._find_state_variable_with_select(
                context,
                lstm_block_cell.output[6],
                [lstm_block_cell]
            )
        return None

    def _ct_ht_shared_variable_finder(self, context, i):
        if self.lstm_cell_type == RNNUnitType.LSTMBlockCell:
            return None

        lstm_cell = context.cell_match[i]
        ct = lstm_cell.get_op("ct").output[0]
        ht = lstm_cell.get_op("ht").output[0]
        ct_concat = [c for c in self.g.find_output_consumers(ct) if is_tf_concat_op(c)]
        ht_concat = [c for c in self.g.find_output_consumers(ht) if is_tf_concat_op(c)]
        if len(ct_concat) != 1 or len(ht_concat) != 1 or ct_concat[0] != ht_concat[0]:
            logger.debug("failed to find ct-ht concat")
            return None
        ct_ht_shared_output = ct_concat[0].output[0]

        consumers = []
        ct_identity_consumer = lstm_cell.get_op("ct_identity_consumer")
        ht_identity_consumer = lstm_cell.get_op("xh")
        ct_slice = [c for c in ct_identity_consumer.inputs if is_tf_slice_op(c)]
        ht_slice = [c for c in ht_identity_consumer.inputs if is_tf_slice_op(c)]
        if len(ct_slice) != 1 or len(ht_slice) != 1:
            logger.debug("failed to find slice op before identity consumers")
            return None
        consumers.extend([ct_slice[0], ht_slice[0]])

        return self._find_state_variable_with_select(
            context,
            ct_ht_shared_output,
            consumers
        )

    def is_valid(self, context):
        # except for ct, ht or ct_ht, there are at most 2 state variables
        if len(context.loop_properties.state_variables) - \
                len(context.state_variables) > 2:
            return False

        # output is no more than 1
        outputs = context.loop_properties.scan_outputs_exits
        if len(outputs) > 1:
            logger.debug("found %d outputs for lstm: %s", len(outputs), outputs)
            return False
        return True

    def _convert_gates_icfo_to_iofc(self, gates, axis=0):
        # from Tensorflow
        return (np.concatenate((g[0], g[3], g[2], g[1]), axis=axis) for g in gates)

    def _convert_gates_ifco_to_iofc(self, gates, axis=0):
        # from Keras
        return (np.concatenate((g[0], g[3], g[1], g[2]), axis=axis) for g in gates)

    def _process_weights_and_bias_per_layer(self, context, i):
        weights = context.weights[i]

        if context.from_keras:
            wx = weights["w"]
            wh = weights["r"]
            w_dtype = weights["w"].dtype
            hidden_size = int(wx.shape[1] / 4)

            # split bias for each hidden unit
            if weights["bias"] is not None:
                b_r_icfo = weights["bias"] # (4 * num_units,)
                b_dtype = weights["bias"].dtype
                bias_dim = b_r_icfo.shape[0]
                assert int(bias_dim / 4) == hidden_size

                b_r_icfo = np.reshape(b_r_icfo, (1, bias_dim))
                bias_gates = np.split(b_r_icfo, 4, axis=1)
                wb_bias_iofc, = self._convert_gates_ifco_to_iofc([bias_gates], axis=1)
            else:
                bias_dim = 4 * hidden_size
                b_dtype = w_dtype # use w_dtype if bias is not given
                wb_bias_iofc = np.zeros((1, bias_dim), dtype=b_dtype)

        else:
            w_r_icfo = weights["weight"]
            w_dtype = weights["weight"].dtype
            b_r_icfo = weights["bias"]
            b_dtype = weights["bias"].dtype
            ft_bias_scalar = weights["ft_bias"]

            # split bias for each hidden unit
            bias_dim = b_r_icfo.shape[0]
            hidden_size = int(bias_dim / 4)
            b_r_icfo = np.reshape(b_r_icfo, (1, bias_dim)) # (4 * num_units,)
            bias_gates = np.split(b_r_icfo, 4, axis=1)
            bias_gates[2] = np.add(bias_gates[2], ft_bias_scalar)
            wb_bias_iofc, = self._convert_gates_icfo_to_iofc([bias_gates], axis=1)

            [wx, wh] = np.split(w_r_icfo, [-1 * hidden_size])
            assert int(wx.shape[1] / 4) == hidden_size

        # fill Rb with zeros since TF and Keras have Wb bias.
        rb_bias_iofc = np.zeros((1, bias_dim), dtype=b_dtype)
        B = np.concatenate((wb_bias_iofc, rb_bias_iofc), axis=1)
        assert B.shape == (1, 2 * bias_dim)

        w_gates = np.split(wx, 4, axis=1)
        h_gates = np.split(wh, 4, axis=1)
        new_w, new_r = self._convert_gates_ifco_to_iofc([w_gates, h_gates], axis=1) if context.from_keras else \
            self._convert_gates_icfo_to_iofc([w_gates, h_gates], axis=1)

        W_iofc = np.transpose(new_w)
        R_iofc = np.transpose(new_r)
        W = np.array([W_iofc], w_dtype)
        R = np.array([R_iofc], w_dtype)

        return W, R, B

    def _make_constants(self, context, i, W, R, B):
        input_size = W.shape[-1]
        hidden_size = R.shape[-1]

        w_name = utils.make_name("W" + str(i))
        w_node = self.g.make_const(w_name, W, skip_conversion=True)

        r_name = utils.make_name("R" + str(i))
        r_node = self.g.make_const(r_name, R, skip_conversion=True)

        b_name = utils.make_name("B" + str(i))
        b_node = self.g.make_const(b_name, B, skip_conversion=True)

        context.input_size[i] = input_size
        context.hidden_size[i] = hidden_size
        context.onnx_input_ids[i]["W"] = w_node.output[0]
        context.onnx_input_ids[i]["R"] = r_node.output[0]
        context.onnx_input_ids[i]["B"] = b_node.output[0]

    def process_weights_and_bias_per_layer(self, context, i):
        W, R, B = self._process_weights_and_bias_per_layer(context, i)
        self._make_constants(context, i, W, R, B) # create node

    def process_weights_and_bias(self, context):
        for i in range(self.num_lstm_layers):
            self.process_weights_and_bias_per_layer(context, i)

    def process_var_init_nodes(self, context):
        for i in range(self.num_lstm_layers):
            self.process_var_init_nodes_per_layer(context, i)

    def process_var_init_nodes_per_layer(self, context, i):
        init_h_id = None
        init_c_id = None
        if "ct_ht" + str(i) in context.state_variables:
            init_h_id, init_c_id = self._process_non_tuple_ch_init_nodes(context, i)
        elif "ct" + str(i) in context.state_variables and ("ht" + str(i)) in context.state_variables:
            init_h_id, init_c_id = self._process_tuple_ch_init_nodes(context, i)
        else:
            raise ValueError("no initializers, unexpected")
        assert init_h_id and init_c_id
        context.onnx_input_ids[i]["initial_h"] = init_h_id
        context.onnx_input_ids[i]["initial_c"] = init_c_id

    def _process_non_tuple_ch_init_nodes(self, context, i):
        gb = GraphBuilder(self.g)
        input_id = context.state_variables["ct_ht" + str(i)].enter_input_id
        hidden_size = context.hidden_size[i]

        attr = {"axes": [1], "starts": [0], "ends": [hidden_size]}
        inputs_map = {"data": input_id, **attr}
        slice_node1 = GraphBuilder(self.g).make_slice(inputs_map)
        unsqueeze_node_1 = gb.make_unsqueeze({'data': slice_node1, "axes": [0]}, return_node=True)

        attr = {"axes": [1], "starts": [hidden_size], "ends": [hidden_size * 2]}
        inputs_map = {"data": input_id, **attr}
        slice_node2 = GraphBuilder(self.g).make_slice(inputs_map)
        unsqueeze_node_2 = gb.make_unsqueeze({'data': slice_node2, "axes": [0]}, return_node=True)

        return unsqueeze_node_1.output[0], unsqueeze_node_2.output[0]

    def _process_tuple_ch_init_nodes(self, context, i):
        h_init_input_id = context.state_variables["ht" + str(i)].enter_input_id
        c_init_input_id = context.state_variables["ct" + str(i)].enter_input_id
        h_node_output = self._process_c_or_h_init_nodes(h_init_input_id, context)
        c_node_output = self._process_c_or_h_init_nodes(c_init_input_id, context)
        return h_node_output, c_node_output

    def _process_c_or_h_init_nodes(self, initializer_input_id, context):
        node = self.g.get_node_by_output(initializer_input_id)
        if node.is_const():
            val = node.get_tensor_value(as_list=False)
            initial_name = utils.make_name("Const")
            new_val = np.expand_dims(val, axis=0)
            const_node = self.g.make_const(initial_name, new_val)
            return const_node.output[0]

        gb = GraphBuilder(self.g)
        squeeze_node = gb.make_unsqueeze({'data': initializer_input_id, "axes": [0]}, return_node=True)
        to_replace = [n for n in self.g.get_nodes() if n != squeeze_node]
        self.g.replace_all_inputs(initializer_input_id, squeeze_node.output[0], ops=to_replace)
        return squeeze_node.output[0]

    def create_single_rnn_node(self, context, i):
        # specify if the RNN is forward, reverse, or bidirectional.
        # Must be one of forward (default), reverse, or bidirectional.
        # Here we won't mark bidirectional/reverse, we will have another rewriter running
        # after this one, which will based on patterns to combine a forward LSTM and a
        # backward LSTM into a bidirectional one.
        num_direction = 1
        context.attributes[i]["direction"] = "forward"
        context.attributes[i]["hidden_size"] = context.hidden_size[i]
        inputs = context.onnx_input_ids[i]
        lstm_inputs = [
            inputs["X"], inputs["W"], inputs["R"], inputs["B"],
            inputs["sequence_lens"], inputs["initial_h"], inputs["initial_c"]]

        x_shape = self.g.get_shape(lstm_inputs[0])
        x_seq_length = x_shape[0]
        x_batch_size = x_shape[1]
        out_dtype = self.g.get_dtype(lstm_inputs[0])

        lstm_node = self.g.make_node("LSTM", lstm_inputs, attr=context.attributes[i], output_count=3,
                                     shapes=[[x_seq_length, num_direction, x_batch_size, context.hidden_size[i]],
                                             [num_direction, x_batch_size, context.hidden_size[i]],
                                             [num_direction, x_batch_size, context.hidden_size[i]]],
                                     dtypes=[out_dtype, out_dtype, out_dtype], op_name_scope=context.rnn_scope)
        return lstm_node

    def create_rnn_node(self, context):
        gb = GraphBuilder(self.g)
        rnn_nodes = list()
        outputs = context.loop_properties.scan_outputs_exits
        logger.debug("number of rnn node outputs: %s", len(outputs))

        for i in range(self.num_lstm_layers):
            logger.debug("creating rnn node for layer: %s", i)
            rnn_nodes.append(self.create_single_rnn_node(context, i))
            output_id = rnn_nodes[i].output[0]
            rnn_output_shape = self.g.get_shape(output_id)
            squeeze_output_shape = [rnn_output_shape[0], rnn_output_shape[2], rnn_output_shape[3]]
            squeeze_node = gb.make_squeeze({"data": output_id, "axes": [1]},
                                           shapes=[squeeze_output_shape],
                                           dtypes=[self.g.get_dtype(output_id)],
                                           return_node=True)
            if i + 1 < self.num_lstm_layers:
                logger.debug("setting input for layer: %s", i + 1)
                context.onnx_input_ids[i + 1]["X"] = squeeze_node.output[0]
        return rnn_nodes

    def _connect_lstm_yh_to_graph(self, context, i):
        # in tf, y_h output shape is: [batch, hidden]
        # in onnx, output shape is: [number_directions, batch, hidden]
        gb = GraphBuilder(self.g)
        exit_output = context.state_variables["ht" + str(i)].exit_output
        output_id = context.rnn_node[i].output[1]
        lstm_yh_shape = self.g.get_shape(output_id)
        squeeze_node = gb.make_squeeze({"data": output_id, "axes": [0]},
                                       shapes=[[lstm_yh_shape[1], lstm_yh_shape[2]]],
                                       dtypes=[self.g.get_dtype(output_id)],
                                       return_node=True)

        self.g.replace_all_inputs(exit_output.id, squeeze_node.output[0])  # ops=self.g.get_nodes()

    def _connect_lstm_yc_to_graph(self, context, i):
        # in tf, y_c output shape is: [batch, hidden]
        # in onnx, output shape is: [number_directions, batch, hidden]
        gb = GraphBuilder(self.g)
        exit_output = context.state_variables["ct" + str(i)].exit_output
        output_id = context.rnn_node[i].output[2]
        lstm_yc_shape = self.g.get_shape(output_id)
        squeeze_node = gb.make_squeeze({"data": output_id, "axes": [0]},
                                       shapes=[[lstm_yc_shape[1], lstm_yc_shape[2]]],
                                       dtypes=[self.g.get_dtype(output_id)],
                                       return_node=True)

        self.g.replace_all_inputs(exit_output.id, squeeze_node.output[0])  # ops=self.g.get_nodes()

    def _connect_lstm_ych_to_graph(self, context, i):
        # in tf, concat of y_c and y_h output shape is: [batch, hidden *2]
        # in onnx, y_c/y_h output shape is: [number_directions, batch, hidden]
        gb = GraphBuilder(self.g)
        exit_output = context.state_variables["ct_ht" + str(i)].exit_output
        lstm_node = context.rnn_node[i]
        yc_shape = self.g.get_shape(lstm_node.output[2])
        concat_output_shape = [yc_shape[0], yc_shape[1], yc_shape[2] * 2]
        concat = self.g.make_node("Concat", [lstm_node.output[2], lstm_node.output[1]],
                                  attr={"axis": 2}, shapes=[concat_output_shape],
                                  dtypes=[self.g.get_dtype(lstm_node.output[2])])

        squeeze_output_shape = [concat_output_shape[1], concat_output_shape[2]]
        squeeze_node = gb.make_squeeze({'data': concat.output[0], "axes": [0]},
                                       shapes=[squeeze_output_shape],
                                       dtypes=[self.g.get_dtype(concat.output[0])],
                                       return_node=True)

        self.g.replace_all_inputs(exit_output.id, squeeze_node.output[0])  # ops=self.g.get_nodes()
