# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.loop_rewriter_base
"""

import logging
from collections import OrderedDict
from tf2onnx import utils
from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher
from tf2onnx.utils import is_tf_loopcond_op, is_tf_tensor_array_op
from tf2onnx.utils import is_tf_tensor_array_gather_op, is_tf_tensor_array_write_op, is_tf_tensor_array_read_op
from tf2onnx.rewriter.rnn_utils import REWRITER_RESULT
from tf2onnx.utils import TensorValueInfo


logger = logging.getLogger(__name__)
INVALID_INPUT_ID = utils.make_name("invalid_input_id")

# todo(pengwa) remove protected-access with changes to Graph/Node later.
# pylint: disable=missing-docstring,invalid-name,unused-argument,using-constant-test,protected-access


class Context(object):
    def __init__(self):
        self.while_context_scope = None
        self.loop_properties = LoopProperties()
        self.loop_cond = None

        self.cell_graph = None  # GraphInfo of cell graph
        self.cond_graph = None  # GraphInfo of condition graph


class GraphInfo(object):
    def __init__(self, ops, inputs, outputs):
        self.nodes = ops
        self.inputs = inputs  # list of TensorValueInfo in order
        self.outputs = outputs  # list of TensorValueInfo in order
        self.dependent_vars = None


class LoopProperties(object):
    def __init__(self):
        # use enter name as key, they are initial inputs.
        # we don't use enter_input_id because it might be
        # used as initial input for more than one Enter nodes.
        self.state_variables = OrderedDict()
        self.scan_variables = OrderedDict()
        self.unneeded_scan_variables = OrderedDict()

        self.tensor_array_inputs = []  # list of type InputTensorArray

    def add_variable(self, var):
        key = (var.enter_name, var.merge_name)
        utils.make_sure(key not in self.scan_variables,
                        "variable %r already exists as scan variable.", key)
        utils.make_sure(key not in self.state_variables,
                        "variable %r already exists as state variable.", key)
        if var.tensor_array_type == TensorArrayVariableType.READ_LAST:
            # If the variable just returns the last value of the constructed tensor array, it doesn't need to be
            # a scan output
            self.unneeded_scan_variables[key] = var
        elif var.tensor_array_type == TensorArrayVariableType.GATHER_ALL:
            self.scan_variables[key] = var
        else:
            self.state_variables[key] = var

    def get_variables(self, checker):
        if not checker:
            return self.all_variables.values()
        return [v for v in self.all_variables.values() if checker(v)]

    @property
    def all_variables(self):
        items = self.state_variables.copy()
        items.update(self.scan_variables)
        items.update(self.unneeded_scan_variables)
        return items

    # state inputs and outputs are in pairs, even though some outputs are not depending on corresponding input,
    # we leave the input id be None.
    @property
    def state_inputs(self):
        return [v.switch_true_identity_output for v in self.state_variables.values()]

    @property
    def state_inputs_initial_values(self):
        return [v.enter_input_id for v in self.state_variables.values()]

    @property
    def state_outputs(self):
        return [v.next_iteration_input for v in self.state_variables.values()]

    @property
    def state_outputs_exits(self):
        return [v.exit_output for v in self.state_variables.values()]

    # scan output (e.g. tensor array) won't be used by next iteration calculation
    @property
    def scan_outputs(self):
        return [v.next_iteration_input for v in self.scan_variables.values()]

    @property
    def scan_outputs_exits(self):
        return [v.exit_output for v in self.scan_variables.values()]

    # treat input tensor array as scan inputs
    def add_scan_input(self, input_tensor_array):
        self.tensor_array_inputs.append(input_tensor_array)

    # usually it is called TensorArrayReadV3
    @property
    def scan_inputs(self):
        return [i.consumer for i in self.tensor_array_inputs]

    @property
    def scan_inputs_initial_values(self):
        return [i.data_input_id for i in self.tensor_array_inputs]

    def has_variable_with_ta_type(self, tensor_array_type):
        for variable in self.all_variables.values():
            if variable.tensor_array_type == tensor_array_type:
                return True
        return False

class TensorArrayVariableType:
    GATHER_ALL = "GATHER_ALL"
    READ_LAST = "READ_LAST"

class LoopVariable(object):
    """In TensorFlow loop, all loop variables are listed both in iteration body graph's inputs, and outputs.
       Loop (state variable 1, state variable 2) {
           # do the calculation
           # updated state variable 1 not necessarily only depends on state variable 1, it might depend
           # on 0, 1 or more state variables.
           # So if it depends on 0 state variable, then switch_true_identity_output.id is None. For this case,
           # during conversion, a fake input for ONNX Loop body graph is created, but not consumed by any node.
           return (updated) state variable 1, (updated) state variable 2, scan variable 1, scan variable 2
       }

       Here we take the perspective of body graph's outputs:
           1. start from the iteration body graph's output (e.g. next_iteration_input.id)
           2. find body graph generating it (those node between NextIteration and Switch)
           3. find the variable initial value (e.g. enter_input_id)
           4. check whether it is a tensor array
           5. the body graph output might go to next iteration as corresponding input
              (e.g. switch_true_identity_output.id).
    """
    def __init__(self, enter_name, merge_name, enter_input_id, next_iteration_input_id,
                 switch_true_identity_output_id, exit_output_id, tensor_array_type, ta_index_id, g):
        self.enter_name = enter_name
        self.merge_name = merge_name
        self.enter_input_id = enter_input_id

        # the output of iteration body graph for this variable
        # should not be None
        utils.make_sure(next_iteration_input_id, "next_iteration_input_id should not be None")
        self.next_iteration_input = TensorValueInfo(next_iteration_input_id, g)

        # the starting point of iteration body graph,
        # might be None when this variable value (either initial value or last iteration output value)
        # is not consumed iteration body graph nodes.
        self.switch_true_identity_output = TensorValueInfo(switch_true_identity_output_id, g)

        # the switch_false branch is ended with Exit, which is a boundary for the loop,
        # might be None when no consumers for the variable output.
        self.exit_output = TensorValueInfo(exit_output_id, g)

        # only applicable for tensor array variable
        self.tensor_array_type = tensor_array_type
        # todo: need check ta's index variable is a scalar starting from 1, and increase by 1 each iteration.
        # then we can be sure this is equivalent to scan output behavior.
        self.ta_index_id = ta_index_id


class InputTensorArray(object):
    def __init__(self, data_input_id, index_input_id, consumer_id, g):
        self.index_input_id = index_input_id
        self.data_input_id = data_input_id

        # tensor array is unstacked before being used in loop, consumer_id is the node
        # (in the iteration body graph) consuming one of the element of tensor array.
        self.consumer = TensorValueInfo(consumer_id, g)


class LoopRewriterBase(object):
    def __init__(self, g):
        self.g = g
        self.ta_read_input_pattern = \
            OpTypePattern("TensorArrayReadV3", name="ta_read", inputs=[
                OpTypePattern("Enter", name="ta_enter", inputs=[
                    OpTypePattern("TensorArrayV3")
                ]),
                OpTypePattern("Identity", name="ta_index"),
                OpTypePattern("Enter", name="ta_scatter_enter", inputs=[
                    OpTypePattern("TensorArrayScatterV3", name="ta_input_scatter")
                ]),
            ])

    def create_context(self):
        return Context()

    def need_rewrite(self, context):
        return False

    def rewrite(self, context):
        return REWRITER_RESULT.FAIL

    def run_internal(self, allow_ta_read_last=False):
        loopcond_ops = []
        for op in self.g.get_nodes():
            if is_tf_loopcond_op(op):
                loopcond_ops.append(op)

        # self.g.get_nodes may change inside this loop so that we parse all LoopCond first
        for op in loopcond_ops:
            logger.debug("======================\n handling loop cond node called %s", op.name)
            context = self.create_context()
            context.loop_cond = op

            self._check_in_read_only_mode(context)  # parses loop variables

            loop_properties = context.loop_properties
            if not allow_ta_read_last and loop_properties.has_variable_with_ta_type(TensorArrayVariableType.READ_LAST):
                continue

            if self.need_rewrite(context):
                # cut off connection between cell/cond graphs and useless nodes like Merge, NextIteration.
                self._cut_off_connection_for_cell(context)
                context.cell_graph = self._crop_loop_body_sub_graph(context)
                context.cond_graph = self._crop_loop_condition_sub_graph(context)

                _result = self.rewrite(context)
                if _result == REWRITER_RESULT.OK:
                    logger.debug("rewrite successfully")
                elif _result == REWRITER_RESULT.SKIP:
                    logger.debug("rewrite skipped for LoopCond called %s", op.name)
                    continue
                elif _result == REWRITER_RESULT.FAIL:
                    raise ValueError("rewrite failed, so just fast fail it")

        if self.g.outputs:
            # clean the graph based on output names.
            self.g.delete_unused_nodes(self.g.outputs)
        return self.g.get_nodes()

    def _check_in_read_only_mode(self, context):
        self._parse_loop_variables(context)
        self._parse_input_ta(context)

    def _parse_loop_variables(self, context):
        loop_cond_op = context.loop_cond
        parts = loop_cond_op.name.split('/')
        context.while_context_scope = '/'.join(parts[0:-1]) + "/"
        logger.debug("found while loop scope %s", context.while_context_scope)

        switch_nodes = self.g.find_output_consumers(loop_cond_op.output[0])
        for s in switch_nodes:
            if s.type != 'Switch':
                raise ValueError("LoopCond's output node should be followed with a Switch node")

            loop_var = self._get_loop_var_from_switch(s)
            context.loop_properties.add_variable(loop_var)

        def inputs_equal(inp1, inp2):
            # Checks input equality with an exception for a Select pattern in some LSTM nodes
            if inp1 == inp2:
                return True
            node1 = self.g.get_node_by_output(inp1)
            node2 = self.g.get_node_by_output(inp2)
            if node1.type != "Select" or node2.type != "Select":
                return False
            if node1.inputs[0].type != "Tile" or node2.inputs[0].type != "Tile":
                return False
            if node1.inputs[0].input[0] != node2.inputs[0].input[0]:
                return False
            # Ignore the tile input. It gets its shape from different nodes but is actually the same.
            return node1.input[1:] == node2.input[1:]

        for unneeded_scan_variable in context.loop_properties.unneeded_scan_variables.values():
            for state_variable in context.loop_properties.state_variables.values():
                if inputs_equal(unneeded_scan_variable.next_iteration_input.id, state_variable.next_iteration_input.id):
                    unneeded_scan_variable.equivalent_state_variable = state_variable
                    break

    def _parse_input_ta(self, context):
        graph_inputs = [v.switch_true_identity_output.id for v in context.loop_properties.all_variables.values()
                        if v.switch_true_identity_output.id]
        matcher = GraphMatcher(self.ta_read_input_pattern, allow_reorder=False)
        match_results = matcher.match_ops(self.g.get_nodes())
        match_results = [r for r in match_results if r.get_op("ta_index").output[0] in graph_inputs]
        for match in match_results:
            ta_input_scatter = match.get_op("ta_input_scatter")
            # the 3rd input of scatter is the value
            data_input_id = ta_input_scatter.input[2]
            ta_read_node = match.get_op("ta_read")

            # todo: need check ta's index variable is a scalar starting from 1, and increase by 1 each iteration.
            # then we can be sure this is equivalent to scan input behavior.
            index_input_id = ta_read_node.input[1]
            unstacked_ta_consumer = match.get_op("ta_read").output[0]
            ta = InputTensorArray(data_input_id, index_input_id, unstacked_ta_consumer, self.g)
            context.loop_properties.add_scan_input(ta)

    def _crop_loop_body_sub_graph(self, context):
        # according to input and output, find the body graph
        loop_props = context.loop_properties
        inputs = loop_props.state_inputs + loop_props.scan_inputs
        input_ids = [input_tensor_value_info.id for input_tensor_value_info in inputs]

        outputs = loop_props.state_outputs + loop_props.scan_outputs
        output_ids = [out_tensor_value_info.id for out_tensor_value_info in outputs]
        ops, enter_nodes, _ = self.find_subgraph(set(input_ids), set(output_ids), self.g, merge_as_end=False)

        for enter_node in enter_nodes:
            # connect Enter's output to Enter's input
            self.g.replace_all_inputs(enter_node.output[0], enter_node.input[0], ops=ops)

        return GraphInfo(ops, inputs, outputs)

    def _crop_loop_condition_sub_graph(self, context):
        input_ids = []
        output_ids = [context.loop_cond.input[0]]
        outputs = [TensorValueInfo(o, self.g) for o in output_ids]
        ops, enter_nodes, merge_nodes = self.find_subgraph(set(input_ids), set(output_ids), self.g, merge_as_end=True)

        for enter_node in enter_nodes:
            # connect Enter's output to Enter's input
            self.g.replace_all_inputs(enter_node.output[0], enter_node.input[0], ops=ops)

        dependent_vars = []
        for merge_node in merge_nodes:
            enter_node = [n for n in merge_node.inputs if n.type == "Enter"][0]
            loop_var = context.loop_properties.all_variables[(enter_node.name, merge_node.name)]

            # cut off connection between condition graph and Merge node.
            # replace condition graph's inputs to be cell graph's outputs, because we want condition graph
            # to consumer cell graph outputs.
            non_switch_consumers = [n for n in self.g.find_output_consumers(merge_node.output[0]) if n.type != "Switch"]
            self.g.replace_all_inputs(merge_node.output[0], loop_var.next_iteration_input.id,
                                      ops=non_switch_consumers)
            dependent_vars.append(loop_var)

        # cut off connection between condition graph and LoopCond node.
        self.g.replace_all_inputs(context.loop_cond.output[0], INVALID_INPUT_ID, ops=[context.loop_cond])

        graph_info = GraphInfo(ops, [], outputs)
        graph_info.dependent_vars = dependent_vars
        return graph_info

    def _cut_off_connection_for_cell(self, context):
        for val in context.loop_properties.all_variables.values():
            if val.switch_true_identity_output.id:
                # remove the node to cut off a starting node of the cell (e.g. loop body).
                n = self.g.get_node_by_output(val.switch_true_identity_output.id)
                self.g.remove_node(n.name)

            if val.tensor_array_type == TensorArrayVariableType.GATHER_ALL:
                # connect NextIteration to an invalid node, to cut off an ending node of the cell.
                ta_write_nodes = [n for n in self.g.get_nodes() if is_tf_tensor_array_write_op(n)]
                self.g.replace_all_inputs(val.next_iteration_input.id, INVALID_INPUT_ID, ops=ta_write_nodes)
            else:
                # connect NextIteration to an invalid node, to cut off an ending node of the cell.
                next_iter_nodes = [n for n in self.g.get_nodes() if n.type == "NextIteration"]
                self.g.replace_all_inputs(val.next_iteration_input.id, INVALID_INPUT_ID, ops=next_iter_nodes)

        for scan_input in context.loop_properties.scan_inputs:
            # remove the node to cut off connection between scan_input and the cell.
            self.g.remove_node(self.g.get_node_by_output(scan_input.id).name)

    def _get_loop_var_from_switch(self, switch_node):
        if switch_node.type != 'Switch':
            logger.error("not a switch node, skip")
            return None

        # the first input is data
        merge_node = switch_node.inputs[0]
        if merge_node.type != "Merge":
            logger.error("switch node does not has Merge as its first input")
            return None

        # find the output_true consumers
        switch_consumers = self.g.find_output_consumers(switch_node.output[1])
        switch_true_consumer_cnt = len(switch_consumers)
        if switch_true_consumer_cnt == 0:
            switch_true_identity_output = None
        elif switch_true_consumer_cnt == 1:
            if switch_consumers[0].type == "Identity":
                switch_true_identity_output = switch_consumers[0].output[0]
            else:
                # using grappler there is not necessarily an identity behind switch
                switch_true_identity_output = switch_node.output[1]
        else:
            # insert identity if there are 2 or more consumers. This can happen on tf-1.15.
            switch_true_identity_output = self.g.make_node("Identity", [switch_node.output[1]],
                                                           shapes=[switch_node.output_shapes[1]],
                                                           dtypes=[switch_node.output_dtypes[1]])
            switch_true_identity_output = switch_true_identity_output.output[0]
            for n in switch_consumers:
                for i, nn in enumerate(n.input):
                    if nn == switch_node.output[1]:
                        n.input[i] = switch_true_identity_output

        target_node_input_id = None
        enter_node = [n for n in merge_node.inputs if n.type == 'Enter'][0]
        target_node_input_id = enter_node.input[0]
        logger.debug("a Switch >> Merge >> Enter is found called %s", enter_node.inputs[0].name)

        next_iteration_node = [n for n in merge_node.inputs if n.type == 'NextIteration'][0]
        last_iteration_output_id = next_iteration_node.input[0]

        # find the output_false consumers to see whether there is consumer for this var
        switch_false_consumers = self.g.find_output_consumers(switch_node.output[0])
        false_consumer_count = len(switch_false_consumers)
        exit_output_id = None
        if false_consumer_count == 1:
            exit_node = switch_false_consumers[0]
            if exit_node.type != "Exit":
                raise ValueError("switch false branch is followed by non-Exit")
            exit_output_id = exit_node.output[0]
        elif false_consumer_count == 0:
            # sometime, the variable output won't be used in the new iteration as input.
            exit_output_id = None
        else:
            raise ValueError("unexpected number of switch false consumers")

        ta_type = None
        ta_index_id = None
        if is_tf_tensor_array_op(self.g.get_node_by_output(target_node_input_id)):

            ta_write_node = self.g.get_node_by_output(last_iteration_output_id)
            utils.make_sure(is_tf_tensor_array_write_op(ta_write_node), "ta nextiteration is not following ta write op")
            last_iteration_output_id = ta_write_node.input[2]
            ta_index_id = ta_write_node.input[1]

            # here we parse patterns generated by
            # ta.write(), then ta.stack(), because this is the most frequent usage pattern.
            if exit_output_id:
                exit_consumers = self.g.find_output_consumers(exit_output_id)
                ta_access_node = [n for n in exit_consumers if is_tf_tensor_array_gather_op(n) or \
                                                               is_tf_tensor_array_read_op(n)][0]

                if is_tf_tensor_array_read_op(ta_access_node):
                    ta_type = TensorArrayVariableType.READ_LAST
                else:
                    ta_type = TensorArrayVariableType.GATHER_ALL

                # update exit output id, treat the gather output as ta's output
                exit_output_id = ta_access_node.output[0]

        loop_var = LoopVariable(enter_node.name, merge_node.name, target_node_input_id, last_iteration_output_id,
                                switch_true_identity_output, exit_output_id, ta_type, ta_index_id, self.g)

        return loop_var

    @staticmethod
    def find_subgraph(input_ids, output_ids, g, merge_as_end=False):
        logger.debug("input ids %s ", input_ids)
        logger.debug("output ids %s ", output_ids)

        enter_nodes = set()
        merge_nodes = set()

        def find_input_boundary(node):
            if node.type == "Enter":
                enter_nodes.add(node)
                logger.debug("terminate the input search at %s", node.name)
                return False

            if merge_as_end is True and node.type == "Merge":
                merge_nodes.add(node)
                logger.debug("terminate the input search at %s", node.name)
                return False

            if node.is_const():
                logger.debug("terminate search at const node %s", node.name)
                return False

            for o in node.output:
                if o in input_ids:
                    return False
            return True

        nodes = g.extract_sub_graph_nodes(output_ids, input_checker=find_input_boundary)
        return nodes, enter_nodes, merge_nodes

    @staticmethod
    def construct_graph_from_nodes(parent_g, nodes, outputs):
        return utils.construct_graph_from_nodes(
            parent_g,
            nodes,
            [out.id for out in outputs],
            [out.shape for out in outputs],
            [out.dtype for out in outputs]
        )
