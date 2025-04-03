# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.custom_rnn_rewriter - custom rnn support
"""

import logging
import sys
import traceback

from onnx import onnx_pb
import numpy as np

from tf2onnx.graph_builder import GraphBuilder
from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase, Context
from tf2onnx.rewriter.rnn_utils import REWRITER_RESULT, get_rnn_scope_name, parse_rnn_loop
from tf2onnx import utils

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,broad-except,protected-access


class CustomRnnContext(Context):
    def __init__(self):
        super(CustomRnnContext, self).__init__()
        self.rnn_scope = None
        self.time_var = None
        self.iteration_var = None


class CustomRnnRewriter(LoopRewriterBase):
    def create_context(self):
        return CustomRnnContext()

    def run(self):
        logger.debug("enter custom rnn rewriter")
        return self.run_internal()

    def need_rewrite(self, context):
        context.rnn_scope = get_rnn_scope_name(context.while_context_scope)

        res = parse_rnn_loop(self.g, context.loop_properties, context.rnn_scope,
                             context.while_context_scope)
        if not res:
            logger.debug("skip the loop due to parse_rnn_loop failed")
            return False

        time_var, iteration_var = res
        context.time_var = time_var
        context.iteration_var = iteration_var
        logger.debug("time var %s - enter input id (%s) shape: %s, output (%s) shape: %s", time_var.enter_name,
                     time_var.enter_input_id, self.g.get_shape(time_var.enter_input_id),
                     time_var.switch_true_identity_output.id, time_var.switch_true_identity_output.shape)

        return True

    def rewrite(self, context):
        logger.debug("enter rewrite function")
        try:
            scan_props = context.loop_properties

            state_inputs_initial_values = []
            for state_input in scan_props.state_inputs_initial_values:
                if self.g.opset == 8:
                    nodes = self._adapt_scan_sequence_input_or_output("input", state_input, False)
                    state_inputs_initial_values.append(nodes[-1].output[0])
                else:  # since opset 9
                    state_inputs_initial_values.append(state_input)

            scan_inputs_initial_values = []
            scan_length = -1
            for scan_input in scan_props.scan_inputs_initial_values:
                if self.g.opset == 8:
                    nodes = self._adapt_scan_sequence_input_or_output("input", scan_input, False)
                    scan_inputs_initial_values.append(nodes[-1].output[0])
                else:  # since opset 9
                    scan_inputs_initial_values.append(scan_input)
                scan_shape = self.g.get_shape(scan_input)
                if scan_shape is not None and len(scan_shape) > 0:
                    scan_length = scan_shape[0]

            cell_g_info = context.cell_graph
            scan_body_g = LoopRewriterBase.construct_graph_from_nodes(self.g, cell_g_info.nodes, cell_g_info.outputs)
            for input_tensor_info in scan_props.state_inputs:
                scan_body_g.add_graph_input(input_tensor_info.id, input_tensor_info.dtype, input_tensor_info.shape)

            for input_tensor_info in scan_props.scan_inputs:
                scan_body_g.add_graph_input(input_tensor_info.id, input_tensor_info.dtype, input_tensor_info.shape)

            scan_node = self._create_scan_node(context, scan_props,
                                               state_inputs_initial_values + scan_inputs_initial_values,
                                               scan_body_g, scan_length)
            if not scan_node:
                logger.error("failed to create scan node during rewrite")
                return REWRITER_RESULT.FAIL

            self._connect_scan_with_output(context, scan_node)

            return REWRITER_RESULT.OK

        except Exception as ex:
            tb = traceback.format_exc()
            logger.error("custom rnn rewrite failed, due to exception: %s, details:%s", ex, tb)
            return REWRITER_RESULT.FAIL

    def _create_scan_node(self, context, scan_props, init_values, body_g, scan_length):
        logger.debug("create scan node")
        branches = {"body": body_g}
        # reuse original output connection id (e.g. Exit_XXX), so we don't need set shape.
        loop_outputs_shapes = []
        loop_outputs_dtypes = []

        for i, out in enumerate(body_g.outputs):
            loop_outputs_dtypes.append(body_g.get_dtype(out))
            shape = body_g.get_shape(out)
            if i < len(scan_props.state_outputs):
                if self.g.opset == 8:
                    loop_outputs_shapes.append([1] + shape)
                else:
                    loop_outputs_shapes.append(shape)
            else:
                if shape is None:
                    loop_outputs_shapes.append(None)
                elif self.g.opset == 8:
                    # In opset 8, the first dim of a scan output is the batch
                    loop_outputs_shapes.append([1, scan_length] + shape)
                else:
                    loop_outputs_shapes.append([scan_length] + shape)

        if self.g.opset == 8:
            # here we did not give the sequence_length, because
            # current batch size is 1, not original batch size
            # original seq_length will be used by the loop body of Scan op.
            scan_node = self.g.make_node("Scan", [""] + init_values, op_name_scope="custom_rnn_scan",
                                         attr={"num_scan_inputs": len(scan_props.scan_inputs)},
                                         output_count=len(scan_props.state_outputs + scan_props.scan_outputs),
                                         shapes=loop_outputs_shapes, dtypes=loop_outputs_dtypes,
                                         skip_conversion=False, branches=branches)
        else:
            scan_node = self.g.make_node("Scan", init_values, op_name_scope="custom_rnn_scan",
                                         attr={"num_scan_inputs": len(scan_props.scan_inputs)},
                                         output_count=len(scan_props.state_outputs + scan_props.scan_outputs),
                                         shapes=loop_outputs_shapes, dtypes=loop_outputs_dtypes,
                                         skip_conversion=False, branches=branches)

        return scan_node

    def _connect_scan_with_output(self, context, scan_node):
        logger.debug("connect scan output with the graph")

        index = 0
        for out_tensor_value_info in context.loop_properties.state_outputs_exits:
            if out_tensor_value_info.id:
                if self.g.opset == 8:
                    nodes = self._adapt_scan_sequence_input_or_output("state_output_reshape",
                                                                      scan_node.output[index], True)
                    self.g.replace_all_inputs(
                        out_tensor_value_info.id, nodes[-1].output[0])  # ops=self.g.get_nodes()
                else:  # since opset 9
                    self.g.replace_all_inputs(
                        out_tensor_value_info.id, scan_node.output[index])  # ops=self.g.get_nodes()
            index += 1

        for out_tensor_value_info in context.loop_properties.scan_outputs_exits:
            if out_tensor_value_info.id:
                if self.g.opset == 8:
                    nodes = self._adapt_scan_sequence_input_or_output("scan_output_reshape",
                                                                      scan_node.output[index], True)
                    self.g.replace_all_inputs(
                        out_tensor_value_info.id, nodes[-1].output[0])  # ops=self.g.get_nodes()
                else:  # since opset 9
                    self.g.replace_all_inputs(
                        out_tensor_value_info.id, scan_node.output[index])  # ops=self.g.get_nodes()
            index += 1

    def _adapt_scan_sequence_input_or_output(self, target_name, input_id, handle_output=False):
        nodes_to_add = []
        shape_node = self.g.make_node("Shape", [input_id])
        nodes_to_add.append(shape_node)
        inferred_shape = self.g.get_shape(input_id)
        if handle_output is True:
            # handle output:
            # if required dim values don't contain more than one -1,
            # just use a const for Reshape's shape input.
            if inferred_shape is not None and inferred_shape[1:].count(-1) <= 1:
                new_shape_node = self.g.make_const(utils.make_name(target_name + "_target_shape"),
                                                   np.array(inferred_shape[1:], dtype=np.int64))
                nodes_to_add.append(new_shape_node)
            else:
                # otherwise, get the dim dynamically, e.g. remove the fake batch size (e.g.1)
                # from [1, time, real-batch, ...]
                origin_shape_node = self.g.make_node("Cast", [shape_node.output[0]],
                                                     {"to": onnx_pb.TensorProto.FLOAT})
                nodes_to_add.append(origin_shape_node)

                attr = {"axes": [0], "starts": [1], "ends": [sys.maxsize]}
                inputs_map = {"data": origin_shape_node.output[0], **attr}
                sliced_shape_node = GraphBuilder(self.g).make_slice(inputs_map)
                nodes_to_add.append(self.g.get_node_by_output(sliced_shape_node))

                new_shape_node = self.g.make_node("Cast", [sliced_shape_node],
                                                  {"to": onnx_pb.TensorProto.INT64})
                nodes_to_add.append(new_shape_node)

            new_shape = inferred_shape[1:]
        else:
            # handle input:
            if inferred_shape is not None and inferred_shape.count(-1) <= 1:
                new_shape_node = self.g.make_const(utils.make_name(target_name + "_target_shape"),
                                                   np.array([1] + inferred_shape, dtype=np.int64))
                nodes_to_add.append(new_shape_node)
            else:
                # add a fake batch size : 1
                fake_batch_size_node = self.g.make_const(utils.make_name(target_name + "_target_shape"),
                                                         np.array([1], dtype=np.int64))
                nodes_to_add.append(fake_batch_size_node)
                new_shape_node = self.g.make_node("Concat",
                                                  [fake_batch_size_node.output[0], shape_node.output[0]],
                                                  attr={"axis": 0})
                nodes_to_add.append(new_shape_node)
            new_shape = [1] + inferred_shape

        reshape_node = self.g.make_node("Reshape", [input_id, new_shape_node.output[0]],
                                        shapes=[new_shape],
                                        dtypes=[self.g.get_dtype(input_id)],
                                        op_name_scope=target_name)
        nodes_to_add.append(reshape_node)
        logger.debug("create Reshape for scan output %s, with output shape %s",
                     reshape_node.output[0], new_shape)
        return nodes_to_add
