# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.loop_rewriter - generic loop support
"""

import logging
import sys
import traceback

from onnx import TensorProto
import numpy as np

from tf2onnx.rewriter.loop_rewriter_base import LoopRewriterBase, Context
from tf2onnx.rewriter.rnn_utils import REWRITER_RESULT
from tf2onnx import utils
from tf2onnx.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,broad-except,protected-access


class LoopRewriter(LoopRewriterBase):

    def create_context(self):
        return Context()

    def run(self):
        logger.debug("enter loop rewriter")
        return self.run_internal(allow_ta_read_last=True)

    def need_rewrite(self, context):
        return True

    def rewrite(self, context):
        logger.debug("enter rewrite function")
        loop_node = None
        try:
            loop_props = context.loop_properties
            cell_g_info = context.cell_graph
            cond_g_info = context.cond_graph

            # create a dummy loop to calculate the init condition
            init_cond_output = self._create_subgraph_initial_cond(cond_g_info)

            ## create Loop body graph with existing nodes

            body_nodes = set(cell_g_info.nodes + cond_g_info.nodes)
            body_outputs = cond_g_info.outputs + cell_g_info.outputs
            for out_tensor_value_info in body_outputs:
                shape = out_tensor_value_info.shape
                utils.make_sure(
                    shape is not None,
                    "Conversion of Loop requries output shape [{}] exists".format(out_tensor_value_info.id)
                )
                out_tensor_value_info.shape = utils.create_vague_shape_like(shape)

            loop_body_g = LoopRewriterBase.construct_graph_from_nodes(self.g, body_nodes, body_outputs)

            # create loop body graph inputs
            loop_body_g.add_graph_input(utils.make_name("i"), TensorProto.INT64, ())
            loop_body_g.add_graph_input(utils.make_name("cond"), TensorProto.BOOL, ())
            for i, tensor_value_info in enumerate(loop_props.state_inputs):
                input_name = tensor_value_info.id
                if input_name is None:
                    # if the variable is not used in the body graph, then we created a fake one,
                    # the same type and shape as its corresponding output.
                    out_tensor_value_info = loop_props.state_outputs[i]
                    dtype = out_tensor_value_info.dtype
                    shape = out_tensor_value_info.shape
                    input_name = utils.make_name("unused_state_input_")
                else:
                    dtype = tensor_value_info.dtype
                    shape = tensor_value_info.shape

                loop_body_g.add_graph_input(input_name, dtype, utils.create_vague_shape_like(shape))

            for input_ta in loop_props.tensor_array_inputs:
                # Loop does not have scan inputs, so we use Gather to get data for each iteration.
                gb = GraphBuilder(loop_body_g)
                index_node = gb.make_unsqueeze({'data': input_ta.index_input_id, "axes": [0]}, return_node=True)
                gather_node = loop_body_g.make_node("Gather", [input_ta.data_input_id, index_node.output[0]])
                data_node = gb.make_squeeze({'data': gather_node.output[0], "axes": [0]}, return_node=True)
                loop_body_g.replace_all_inputs(input_ta.consumer.id, data_node.output[0])  # ops=loop_body_g.get_nodes()

            ## create Loop node
            branches = {"body": loop_body_g}
            loop_node = self._create_loop_node(context, loop_props, init_cond_output, branches=branches)
            if not loop_node:
                logger.error("failed to create loop node during rewrite")
                return REWRITER_RESULT.FAIL

            for unneeded_scan_variable in loop_props.unneeded_scan_variables.values():
                self.g.replace_all_inputs(unneeded_scan_variable.exit_output.id,
                                          unneeded_scan_variable.equivalent_state_variable.exit_output.id)

            logger.debug("rewrite successfully")
            return REWRITER_RESULT.OK

        except Exception as ex:
            tb = traceback.format_exc()
            logger.error("loop rewrite failed, due to exception: %s, details:%s", ex, tb)
            return REWRITER_RESULT.FAIL

    def _create_subgraph_initial_cond(self, cond_graph):
        """Create subgraph to calculate initial cond."""
        # copy condition subgraph to parent graph
        copied_nodes = []
        name_scope = utils.make_name("copy")
        for node in cond_graph.nodes:
            new_name = "{}/{}".format(name_scope, node.name)
            new_outputs = ["{}/{}".format(name_scope, out) for out in node.output]
            # some inputs are out of cond_graph.nodes, keep them intact
            new_inputs = []
            for inp in node.input:
                if self.g.get_node_by_output(inp) in cond_graph.nodes:
                    new_inputs.append("{}/{}".format(name_scope, inp))
                else:
                    new_inputs.append(inp)

            new_node = self.g.make_node(
                node.type, new_inputs, outputs=new_outputs,
                attr=node.attr, name=new_name,
                shapes=node.output_shapes, dtypes=node.output_dtypes,
                skip_conversion=node.skip_conversion, infer_shape_dtype=False
            )
            body_graphs = node.graph.contained_graphs.pop(node.name, None)
            if body_graphs:
                for attr_name, body_graph in body_graphs.items():
                    body_graph.parent_graph = self.g
                    new_node.set_body_graph_as_attr(attr_name, body_graph)
            copied_nodes.append(new_node)

        # replace all inputs of condition graph by initializer (enter_input)
        for loop_var in cond_graph.dependent_vars:
            self.g.replace_all_inputs(
                loop_var.next_iteration_input.id,
                loop_var.enter_input_id, ops=copied_nodes)
        init_cond_output = "{}/{}".format(name_scope, cond_graph.outputs[0].id)
        self.g.set_dtype(init_cond_output, cond_graph.outputs[0].dtype)
        self.g.set_shape(init_cond_output, cond_graph.outputs[0].shape)
        return init_cond_output

    def _create_loop_node(self, context, loop_props, init_cond_output, branches=None):
        loop_outputs = []
        loop_output_shapes = []
        loop_output_dtypes = []
        for tensor_value_info in loop_props.state_outputs_exits + loop_props.scan_outputs_exits:
            if tensor_value_info.id:
                loop_outputs.append(tensor_value_info.id)
                loop_output_shapes.append(tensor_value_info.shape)
                loop_output_dtypes.append(tensor_value_info.dtype)
                n = self.g.get_node_by_output(tensor_value_info.id)
                self.g.remove_node(n.name)
            else:
                output_name = utils.make_name("unused_loop_output_")
                tensor_value_info.id = output_name
                loop_outputs.append(output_name)
                loop_output_shapes.append([-1])
                loop_output_dtypes.append(None)

        # trip count and cond are not used, giving them values just because bug
        # (https://github.com/Microsoft/onnxruntime/issues/255) of onnxruntime.
        trip_cnt = self.g.make_const(utils.make_name("trip_count"), np.array(sys.maxsize, dtype=np.int64))
        loop_node = self.g.make_node("Loop", [trip_cnt.output[0]] + [init_cond_output] +
                                     loop_props.state_inputs_initial_values,  # ONNX Loop support state inputs only
                                     outputs=loop_outputs, op_name_scope="generic_loop",
                                     shapes=loop_output_shapes, dtypes=loop_output_dtypes,
                                     skip_conversion=False, branches=branches)

        return loop_node
