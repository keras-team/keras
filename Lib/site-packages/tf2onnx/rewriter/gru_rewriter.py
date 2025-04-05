# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.gru_rewriter
"""

import logging
import numpy as np
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.rewriter.rnn_utils import RNNUnitType, get_weights_from_const_node

from tf2onnx.rewriter.unit_rnn_rewriter_base import UnitRnnRewriterBase

# pylint: disable=invalid-name,unused-argument,missing-docstring


logger = logging.getLogger(__name__)


class GRUUnitRewriter(UnitRnnRewriterBase):
    def __init__(self, g):
        super(GRUUnitRewriter, self).__init__(g)
        self.gru_cell_type = None
        self.state_variable_handlers = [
            {"state": (self._state_variable_finder, self._connect_gru_state_to_graph)}
        ]

    def run(self):
        logger.debug("enter gru rewriter")
        return super(GRUUnitRewriter, self).run()

    def find_cell(self, context):
        gru_cell_types = [RNNUnitType.GRUCell, RNNUnitType.GRUBlockCell, RNNUnitType.CudnnCompatibleGRUCell]
        for cell_type in gru_cell_types:
            cell_match = self._match_cell(context, cell_type)
            if cell_match:
                self.gru_cell_type = cell_type
                logger.debug("parsing unit is %s", cell_type)
                return cell_match
        logger.debug("cannot parse unit")
        return None

    def get_weight_and_bias(self, context):
        match = context.cell_match

        gate_kernel = get_weights_from_const_node(self.g, match.get_op("gate_kernel"))
        gate_bias = get_weights_from_const_node(self.g, match.get_op("gate_bias"))
        res = {
            "gate_kernel": gate_kernel,
            "gate_bias": gate_bias
        }

        # differ on memory gate:
        # GRUCell: h'_t = tanh(concat(x_t, r_t .* h_t-1) * W + b)
        # CudnnCompatibleGRUCell: h'_t = tanh(x_t * W_x + b_x + r_t .* (h_t-1 * W_h + b_h))
        if self.gru_cell_type == RNNUnitType.CudnnCompatibleGRUCell:
            hidden_state_kernel = get_weights_from_const_node(
                self.g, match.get_op("hidden_state_kernel")
            )
            hidden_state_bias = get_weights_from_const_node(
                self.g, match.get_op("hidden_state_bias")
            )
            hidden_input_kernel = get_weights_from_const_node(
                self.g, match.get_op("hidden_input_kernel")
            )
            hidden_input_bias = get_weights_from_const_node(
                self.g, match.get_op("hidden_input_bias")
            )
            if not all(val is not None for val in [
                    hidden_state_kernel, hidden_state_bias,
                    hidden_input_kernel, hidden_input_bias
            ]):
                logger.debug("rnn weights check failed, skip")
                return None
            hidden_kernel = np.concatenate([hidden_input_kernel, hidden_state_kernel])
            # apply the linear transformation before multiplying by the output of reset gate
            context.attributes["linear_before_reset"] = 1
            res["hidden_kernel"] = hidden_kernel
            res["hidden_bias"] = hidden_input_bias
            # recurrence bias for hidden gate
            res["Rb_h"] = hidden_state_bias
        elif self.gru_cell_type in [RNNUnitType.GRUCell, RNNUnitType.GRUBlockCell]:
            hidden_kernel = get_weights_from_const_node(self.g, match.get_op("hidden_kernel"))
            hidden_bias = get_weights_from_const_node(self.g, match.get_op("hidden_bias"))
            res["hidden_kernel"] = hidden_kernel
            res["hidden_bias"] = hidden_bias

        if not all(val is not None for val in res.values()):
            logger.debug("rnn weights check failed, skip")
            return None

        logger.debug("find needed weights")
        return res

    def _state_variable_finder(self, context):
        if self.gru_cell_type in [
                RNNUnitType.GRUCell,
                RNNUnitType.CudnnCompatibleGRUCell
        ]:
            gru_cell = context.cell_match
            return self._find_state_variable_with_select(
                context,
                gru_cell.get_op("cell_output").output[0],
                [gru_cell.get_op("cell_inputs")]
            )
        if self.gru_cell_type == RNNUnitType.GRUBlockCell:
            gru_block_cell = context.cell_match.get_op("gru_block_cell")
            return self._find_state_variable_with_select(
                context,
                gru_block_cell.output[3],
                [gru_block_cell]
            )
        return None

    def parse_attributes(self, context):
        # in tf, only activation of hidden gate is optional, input and update gate always use sigmoid
        match = context.cell_match
        activations = ["Sigmoid", "Tanh"]
        if self.gru_cell_type == RNNUnitType.GRUCell:
            activation_op = match.get_op("optional_activation")
            activations = ["Sigmoid", activation_op.type]
        context.attributes["activations"] = activations
        return True

    def is_valid(self, context):
        # except for ct, ht or ct_ht, there are at most 2 state variables
        other_state_variables_num = len(context.loop_properties.state_variables) - \
            len(context.state_variables)
        if other_state_variables_num > 2:
            logger.debug("found %d other state variables", other_state_variables_num)
            return False

        # output should be no more than 1
        outputs = context.loop_properties.scan_outputs_exits
        if len(outputs) > 1:
            logger.debug("found %d outputs for gru: %s", len(outputs), outputs)
            return False
        return True

    def _make_constants(self, context, W_zrh, R_zrh, B_zrh):
        input_size = W_zrh.shape[-1]
        hidden_size = R_zrh.shape[-1]
        w_name = utils.make_name("W")
        w_node = self.g.make_const(w_name, W_zrh, skip_conversion=True)

        r_name = utils.make_name("R")
        r_node = self.g.make_const(r_name, R_zrh, skip_conversion=True)

        b_name = utils.make_name("B")
        b_node = self.g.make_const(b_name, B_zrh, skip_conversion=True)

        context.input_size = input_size
        context.hidden_size = hidden_size
        context.onnx_input_ids["W"] = w_node.output[0]
        context.onnx_input_ids["R"] = r_node.output[0]
        context.onnx_input_ids["B"] = b_node.output[0]

    def _process_weights_and_bias_keras(self, context):
        weights = context.weights
        W_zrh = np.expand_dims(weights["gate_kernel"].transpose(), axis=0)
        R_zrh = np.expand_dims(weights["hidden_kernel"].transpose(), axis=0)
        Wb_zrh = weights["gate_bias"]
        Rb_zrh = weights["hidden_bias"]
        B_zrh = np.expand_dims(np.concatenate((Wb_zrh, Rb_zrh), axis=0), axis=0)
        self._make_constants(context, W_zrh, R_zrh, B_zrh)

    def process_weights_and_bias(self, context):
        """
        why split the data in this way should refer to code of tensorflow GRU cell and official document of ONNX GRU
        """
        if context.from_keras:
            self._process_weights_and_bias_keras(context)
            return
        weights = context.weights
        # from code of tensorflow GRU cell, it can be known that shape of hidden_kernel(or candidate_kernel)
        # is (input_size+hidden_unit, hidden_unit)
        hidden_size = weights["hidden_kernel"].shape[1]
        input_size = weights["hidden_kernel"].shape[0] - hidden_size
        weight_dtype = weights["hidden_kernel"].dtype
        bias_dtype = weights["hidden_bias"].dtype
        # below code will use same notation as ONNX document
        # z means update gate, r means reset gate, h means hidden gate;
        # at this time weights of gate include input and state, will split it next
        r_kernel, z_kernel = np.split(weights["gate_kernel"], [hidden_size], axis=1)
        h_kernel = weights["hidden_kernel"]
        r_bias, z_bias = np.split(weights["gate_bias"], [hidden_size], axis=0)
        h_bias = weights["hidden_bias"]
        for k in sorted(weights.keys()):
            print(k, weights[k].shape)
        # ONNX GRU split weights of input and state, so have to split *_kernel
        input_r_kernel, state_r_kernel = np.split(r_kernel, [input_size], axis=0)
        input_z_kernel, state_z_kernel = np.split(z_kernel, [input_size], axis=0)
        input_h_kernel, state_h_kernel = np.split(h_kernel, [input_size], axis=0)
        W_zrh = np.concatenate((input_z_kernel, input_r_kernel, input_h_kernel), axis=1)
        R_zrh = np.concatenate((state_z_kernel, state_r_kernel, state_h_kernel), axis=1)
        # transpose weight matrix
        W_zrh = np.transpose(np.expand_dims(W_zrh, axis=0), axes=(0, 2, 1))
        R_zrh = np.transpose(np.expand_dims(R_zrh, axis=0), axes=(0, 2, 1))
        W_zrh = W_zrh.astype(weight_dtype)
        R_zrh = R_zrh.astype(weight_dtype)
        assert W_zrh.shape == (1, 3*hidden_size, input_size)
        assert R_zrh.shape == (1, 3*hidden_size, hidden_size)
        Wb_zrh = np.concatenate((z_bias, r_bias, h_bias), axis=0)
        # if tf doesn't provide bias for state, use 0
        zero = np.zeros_like(z_bias)
        # Rb_h is set in CudnnCompatibleGRUCell
        Rb_h = weights["Rb_h"] if "Rb_h" in weights else zero
        Rb_zrh = np.concatenate((zero, zero, Rb_h), axis=0)
        B_zrh = np.concatenate((Wb_zrh, Rb_zrh), axis=0)
        B_zrh = np.expand_dims(B_zrh, axis=0)
        B_zrh = B_zrh.astype(bias_dtype)
        assert B_zrh.shape == (1, 6*hidden_size)
        # create const ONNX node
        self._make_constants(context, W_zrh, R_zrh, B_zrh)

    def process_var_init_nodes(self, context):
        assert "state" in context.state_variables.keys()
        initializer_input_id = context.state_variables["state"].enter_input_id
        node = self.g.get_node_by_output(initializer_input_id)
        if node.is_const():
            val = node.get_tensor_value(as_list=False)
            initial_name = utils.make_name("Const")
            new_val = np.expand_dims(val, axis=0)
            const_node = self.g.make_const(initial_name, new_val)
            context.onnx_input_ids["initial_state"] = const_node.output[0]
            return
        squeeze_node = GraphBuilder(self.g).make_unsqueeze(
            {'data': initializer_input_id, 'axes': [0]}, return_node=True)
        to_replace = [n for n in self.g.get_nodes() if n != squeeze_node]
        self.g.replace_all_inputs(initializer_input_id, squeeze_node.output[0], ops=to_replace)
        context.onnx_input_ids["initial_state"] = squeeze_node.output[0]

    def create_rnn_node(self, context):
        # specify if the RNN is forward, reverse, or bidirectional.
        # Must be one of forward (default), reverse, or bidirectional.
        # Here we won't mark bidirectional/reverse, we will have another rewriter running after this one,
        # which will based on patterns to combine a forward GRU and a backward GRU into a bidirectional one.
        num_direction = 1
        # todo: input_forget
        context.attributes["direction"] = "forward"
        context.attributes["hidden_size"] = context.hidden_size
        inputs = context.onnx_input_ids
        # sequence length is optional
        seq_len_input = utils.ONNX_EMPTY_INPUT
        if inputs["sequence_lens"]:
            seq_len_input = inputs["sequence_lens"]
        gru_inputs = [
            inputs["X"], inputs["W"], inputs["R"], inputs["B"],
            seq_len_input, inputs["initial_state"]]
        x_shape = self.g.get_shape(gru_inputs[0])
        x_seq_length = x_shape[0]
        x_batch_size = x_shape[1]
        out_dtype = self.g.get_dtype(gru_inputs[0])
        gru_node = self.g.make_node("GRU", gru_inputs, attr=context.attributes, output_count=2,
                                    shapes=[[x_seq_length, num_direction, x_batch_size, context.hidden_size],
                                            [num_direction, x_batch_size, context.hidden_size]],
                                    dtypes=[out_dtype, out_dtype], op_name_scope=context.rnn_scope)
        return gru_node

    def _connect_gru_state_to_graph(self, context):
        # in tf, state output shape is: [batch, hidden]
        # in onnx, output shape is: [number_directions, batch, hidden]
        exit_output_id = context.state_variables["state"].exit_output.id
        if not exit_output_id:
            logger.debug("no one consume state variable")
            return
        output_id = context.rnn_node.output[1]
        gru_state_shape = self.g.get_shape(output_id)
        output_shape = [gru_state_shape[1], gru_state_shape[2]]
        squeeze_node = GraphBuilder(self.g).make_squeeze(
            {'data': output_id, "axes": [0]}, shapes=[output_shape],
            dtypes=[self.g.get_dtype(output_id)], return_node=True)
        self.g.replace_all_inputs(exit_output_id, squeeze_node.output[0])  # ops=self.g.get_nodes()
