# SPDX-License-Identifier: Apache-2.0


# Temporary base class exclusive for LSTMs for stacked LSTM layer support.
# Once GRU, BiLSTM, BiGRU re-writers will also be enhanced for stacked layer support
# this will be combined with unit rnn base class.

"""
tf2onnx.rewriter.lstm_rewriter_base
"""

import logging

from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase
from tf2onnx.rewriter.rnn_utils import get_pattern
from tf2onnx.graph_matcher import GraphMatcher
from tf2onnx.rewriter.unit_rnn_rewriter_base import UnitRnnRewriterBase, UnitRnnContext

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,broad-except,protected-access,W0223

class LSTMContext(UnitRnnContext):
    def __init__(self):
        super(LSTMContext, self).__init__()
        self.cell_match = list()  # matched cell

        self.weights = list({})
        self.input_size = list()
        self.hidden_size = list()

        self.attributes = list({})  # onnx attributes
        # onnx inputs: List of [X, W, R, B, sequence_lens, initial_h, initial_c, P]
        self.onnx_input_ids = list({})


class LSTMRewriterBase(UnitRnnRewriterBase):
    """
    main procedures:
    1 check whether extracted loop is a unit LSTM, fall back in necessity:
        1 parse LSTM
        2 find needed info from tensorflow graph
    3 process found info according to ONNX requirement
    """

    def create_context(self):
        return LSTMContext()

    def parse_unit_rnn(self, context):
        """
        parse needed info from tensorflow graph:
        1 weight
        2 state variables used in rnn unit, such as c_t, h_t
        3 sequence node
        4 input_x
        5 attributes, e.g., activation_alpha, activation_beta... optional
        """
        logger.debug("parse unit rnn")
        self.state_variable_handler = list()
        self.state_variable_handlers = list()

        logger.debug("match unit cell against loop body graph")
        cell_match = self.find_cell(context)
        if not cell_match:
            logger.debug('failed to match cell pattern')
            return False
        cell_match.sort(key=lambda cmt: cmt.get_op("cell_kernel").name)
        context.cell_match = cell_match

        logger.debug("get_weight_and_bias starts")
        weights = self.get_weight_and_bias(context)
        if not weights:
            logger.debug("rnn weights check failed, SKIP")
            return False
        context.weights = weights

        if not self.get_state_variables(context):
            logger.debug("no cell variable initializers found, SKIP")
            return False

        seq_len_node = self.find_sequence_length_node(context)
        if seq_len_node:
            logger.debug("find sequence node: %s", seq_len_node.name)

        # require exact one input
        inputs = context.loop_properties.scan_inputs_initial_values
        if len(inputs) != 1:
            logger.debug("found %d inputs for the unit rnn: %s",
                         len(inputs), inputs)
            return False

        for i in range(len(context.cell_match)):
            context.onnx_input_ids.append({})
            context.input_size.append(None)
            context.hidden_size.append(None)
            context.attributes.append({})
            context.onnx_input_ids[i]["sequence_lens"] = \
                seq_len_node.output[0] if seq_len_node else utils.ONNX_EMPTY_INPUT

        context.onnx_input_ids[0]["X"] = inputs[0]
        if not self.parse_attributes(context):
            logger.debug("wrong attributes found")
            return False

        return True

    def _match_cell(self, context, unittype):
        """match unit cell"""
        for cell_pattern in get_pattern(unittype):
            matcher = GraphMatcher(cell_pattern, allow_reorder=True)

            loop_props = context.loop_properties
            inputs = loop_props.state_inputs + loop_props.scan_inputs
            input_ids = [input_tensor_value_info.id for input_tensor_value_info in inputs]
            outputs = loop_props.state_outputs + loop_props.scan_outputs
            output_ids = [out_tensor_value_info.id for out_tensor_value_info in outputs]
            body_graph_ops, _, _ = LoopRewriterBase.find_subgraph(
                set(input_ids),
                set(output_ids),
                self.g, merge_as_end=True
            )

            match_results = list(matcher.match_ops(body_graph_ops))
            logger.debug("number of match results: %s", len(match_results))
            if len(match_results) > 0:
                return match_results
        return None

    def get_state_variables(self, context):
        """
        Get state variables by provided handlers. There maybe several handlers corresponding to
        different patterns of state variables.
        The commone method is to find state variables from loop property according to its
        next_iteration_input and switch_true_identity_output, see lstm_rewriter_v2
        """
        contains_handler = False
        for handler in self.state_variable_handlers:
            can_handle = True
            for var_name, funcs in handler.items():
                finder = funcs[0]
                state_variable = finder(context, funcs[2])
                if state_variable:
                    logger.debug("found state variable %s", var_name)
                    context.state_variables[var_name] = state_variable
                else:
                    logger.debug("failed to get state variable %s", var_name)
                    can_handle = False
                    break
            if can_handle:
                self.state_variable_handler.append(handler)
                contains_handler = True
        return contains_handler

    def process_outputs(self, context):
        for handler in self.state_variable_handler:
            for var_name, funcs in handler.items():
                output_connector = funcs[1]
                output_connector(context, funcs[2])
                logger.debug("connect output of %s to graph", var_name)
        logger.debug("done handling all state variables, now focusing on final output")
        self.connect_unit_rnn_output_to_graph(context)

    def connect_unit_rnn_output_to_graph(self, context):
        outputs = context.loop_properties.scan_outputs_exits
        if not outputs:
            logger.debug("no one consume output")
            return

        gb = GraphBuilder(self.g)
        gather_output_id = outputs[0].id
        logger.debug("found output for rnn: %s", gather_output_id)

        # in tf batch major mode, output shape is : [batch, time, hidden]
        # in time major mode, output shape is: [time, batch, hidden]
        # in onnx, output shape is : [time, num_directions, batch, hidden]

        rnn_node = context.rnn_node[len(context.rnn_node) - 1]
        output_id = rnn_node.output[0]
        rnn_output_shape = self.g.get_shape(output_id)
        squeeze_output_shape = [rnn_output_shape[0], rnn_output_shape[2], rnn_output_shape[3]]
        squeeze_node = gb.make_squeeze({'data': output_id, "axes": [1]},
                                       shapes=[squeeze_output_shape],
                                       dtypes=[self.g.get_dtype(output_id)],
                                       return_node=True)
        self.g.replace_all_inputs(gather_output_id, squeeze_node.output[0])  # ops=self.g.get_nodes()
