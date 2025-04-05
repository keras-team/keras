# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.rewriter.cond_rewriter
"""

import logging
import traceback
from collections import OrderedDict
from enum import Enum
from tf2onnx import utils


logger = logging.getLogger(__name__)


# pylint: disable=missing-docstring,unused-argument,broad-except

class BranchType(Enum):
    """Type of branch"""
    TRUE = 1
    FALSE = 2
    # TODO: sometimes, the branch depends on control inputs,
    # so we just set it unknown
    UNKNOWN = 3


class CondBranchContext:
    """Context for each branch graph"""

    def __init__(self):
        self.output = []
        self.nodes = set()


class CondContext:
    def __init__(self, cond_scope, pred_input, true_branch_context,
                 false_branch_context, switchs, merges):
        self.cond_scope = cond_scope  # name scope for this tf.cond
        self.pred_input = pred_input  # condition input
        self.true_branch_context = true_branch_context
        self.false_branch_context = false_branch_context
        self.switchs = set(switchs)
        self.merges = merges  # list of merges in order


class CondRewriter:
    def __init__(self, g):
        self.g = g

    def rewrite(self):
        logger.debug("enter cond pre rewrite")
        return self.run()

    def run(self):
        """tf.cond rewriter"""
        # parse tf.cond in topological sort order.
        # NOTE: we assume the current graph is a DAG.
        name_scope_merges = OrderedDict()
        self.g.topological_sort(self.g.get_nodes())
        all_nodes = self.g.get_nodes()
        for n in all_nodes:
            if self._is_merge(n):
                name_scope = utils.tf_name_scope(n.name)
                if name_scope not in name_scope_merges:
                    name_scope_merges[name_scope] = []
                name_scope_merges[name_scope].append(n)
        # check if need rewrite
        if not name_scope_merges.keys():
            return all_nodes

        for name_scope, merge_nodes in name_scope_merges.items():
            cond_context = None
            try:
                pred_input, true_branch_context, false_branch_context, switchs = \
                    self._parse_cond(name_scope, merge_nodes)
                cond_context = CondContext(
                    name_scope,
                    pred_input,
                    true_branch_context,
                    false_branch_context,
                    switchs,
                    merge_nodes
                )
            except Exception as ex:
                tb = traceback.format_exc()
                logger.warning("tf.cond rewrite failed, due to exception: %s, details:%s", ex, tb)
                continue

            self._cut_off_connection(cond_context)
            if_node = self._create_if_node(cond_context)
            # remove nodes in If branches explicitly
            if if_node is not None:
                for n in list(cond_context.true_branch_context.nodes) + list(cond_context.false_branch_context.nodes):
                    self.g.remove_node(n.name)
        logger.debug("cond pre rewrite done")

        return self.g.get_nodes()

    def _get_output_shape_dtype(self, cond_context):
        output_shapes = []
        output_dtypes = []
        for i, _ in enumerate(cond_context.true_branch_context.output):
            true_output = cond_context.true_branch_context.output[i]
            false_output = cond_context.false_branch_context.output[i]
            true_shape = self.g.get_shape(true_output)
            utils.make_sure(true_shape is not None, "Shape of {} is None".format(true_output))
            true_rank = len(true_shape)
            true_dtype = self.g.get_dtype(true_output)
            false_shape = self.g.get_shape(false_output)
            utils.make_sure(false_shape is not None, "Shape of {} is None".format(false_output))
            false_rank = len(false_shape)
            false_dtype = self.g.get_dtype(false_output)
            # just require rank is equal
            if true_rank != false_rank:
                raise RuntimeError(
                    "the rank of outputs {} and {} mismatch: {}, {}".format(
                        true_output,
                        false_output,
                        true_rank,
                        false_rank
                    )
                )
            if true_dtype != false_dtype:
                raise RuntimeError(
                    "the dtype of outputs {} and {} mismatch: {}, {}".format(
                        true_output,
                        false_output,
                        true_dtype,
                        false_dtype
                    )
                )
            output_shapes.append(utils.create_vague_shape_like(true_shape))
            output_dtypes.append(true_dtype)
        return output_shapes, output_dtypes

    def _create_if_node(self, cond_context):
        output_shapes, output_dtypes = self._get_output_shape_dtype(cond_context)
        pred_node = self.g.get_node_by_output(cond_context.pred_input)
        while pred_node.type == "Identity":
            pred_node = pred_node.inputs[0]
        if pred_node.is_const():
            # Constant folding for if node
            if pred_node.get_tensor_value():
                branch_outputs = cond_context.true_branch_context.output
            else:
                branch_outputs = cond_context.false_branch_context.output
            for merge, out in zip(cond_context.merges, branch_outputs):
                self.g.replace_all_inputs(merge.output[0], out)
            return None

        true_graph = utils.construct_graph_from_nodes(
            self.g,
            list(cond_context.true_branch_context.nodes),
            cond_context.true_branch_context.output,
            output_shapes,
            output_dtypes
        )
        false_graph = utils.construct_graph_from_nodes(
            self.g,
            list(cond_context.false_branch_context.nodes),
            cond_context.false_branch_context.output,
            output_shapes,
            output_dtypes
        )
        branches = {"then_branch": true_graph, "else_branch": false_graph}
        if_node = self.g.make_node(
            "If",
            [cond_context.pred_input],
            op_name_scope=cond_context.cond_scope,
            outputs=[m.output[0] for m in cond_context.merges],
            shapes=output_shapes,
            dtypes=output_dtypes,
            skip_conversion=False,
            branches=branches
        )
        return if_node

    def _cut_off_connection(self, cond_context):
        """Cut off switchs and merges, all changes are based on the origin graph"""
        nodes_to_add = []
        logger.debug("cut off switch connection")
        # replace switch with identity node
        for switch in cond_context.switchs:
            shapes = switch.output_shapes
            dtypes = switch.output_dtypes
            self.g.remove_node(switch.name)
            false_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[0]],
                op_name_scope=cond_context.cond_scope,
                shapes=[shapes[0]],
                dtypes=[dtypes[0]],
            )
            cond_context.false_branch_context.nodes.add(false_switch_id)
            true_switch_id = self.g.make_node(
                "Identity",
                [switch.input[0]],
                outputs=[switch.output[1]],
                op_name_scope=cond_context.cond_scope,
                shapes=[shapes[1]],
                dtypes=[dtypes[1]],
            )
            cond_context.true_branch_context.nodes.add(true_switch_id)
            nodes_to_add.extend([false_switch_id, true_switch_id])
        # replace merge with if node
        logger.debug("cut off merge connection")
        for n in cond_context.merges:
            self.g.remove_node(n.name)

    def _is_merge(self, node):
        return node.type == "Merge"

    def _is_switch(self, node):
        return node.type == "Switch"

    def _parse_cond(self, name_scope, merge_nodes):
        """Parse condition subgraph for these merge nodes"""
        true_branch_context, false_branch_context, switchs = self._trace_back(name_scope, merge_nodes)
        # find pred output from any switch
        pred_input = list(switchs)[0].input[1]
        return pred_input, true_branch_context, false_branch_context, switchs

    def _trace_back(self, name_scope, merge_nodes):
        """
        Trace back to the switch from merge nodes and collect the nodes
        in the true/false branchs of tf.cond respectively, some comments:
        1. According to tf.cond implementation, We make the hypothesis
           that one tf.cond cannot comprise successive Switch nodes.
        2. Thank to construct_graph_from_nodes, in which Identity node
           will be added to each output of subgraph, we needn't deal with the
           branch with only one const node specially.

        TODO: This implement doesn't depend on control inputs. For a price,
           in the case that true and false branch both only contain a
           const node, we will throw a Exception.
        """
        logger.debug("trace back from [%s]", ",".join(n.name for n in merge_nodes))
        true_branch_context = CondBranchContext()
        false_branch_context = CondBranchContext()
        total_switchs = set()
        for merge_node in merge_nodes:
            true_branch_nodes, true_output, false_branch_nodes, false_output, switchs = \
                self._trace_back_from_one_merge(merge_node)
            true_branch_context.nodes |= set(true_branch_nodes)
            true_branch_context.output.append(true_output)
            false_branch_context.nodes |= set(false_branch_nodes)
            false_branch_context.output.append(false_output)
            total_switchs |= switchs
        return true_branch_context, false_branch_context, total_switchs

    def _trace_back_from_one_merge(self, merge_node):
        """Parse the ingredients (nodes and outputs)of true and false branch"""
        logger.debug("trace back from %s", merge_node.name)
        true_branch_nodes = None
        true_output = None
        false_branch_nodes = None
        false_output = None
        merge_input_1 = merge_node.input[0]
        merge_input_2 = merge_node.input[1]
        switchs = set()

        def stop_at_switch(node):
            if self._is_switch(node):
                switchs.add(node)
                return False
            return True

        branch_nodes_1 = self.g.extract_sub_graph_nodes(
            [merge_input_1],
            stop_at_switch
        )
        branch_nodes_2 = self.g.extract_sub_graph_nodes(
            [merge_input_2],
            stop_at_switch
        )
        branch_type_1 = self._branch_type(merge_input_1, branch_nodes_1)
        branch_type_2 = self._branch_type(merge_input_2, branch_nodes_2)
        # all possible branch types: UU, UT, UF, TU, TF, FU, FT
        if branch_type_1 == BranchType.UNKNOWN and branch_type_2 == BranchType.UNKNOWN:
            raise ValueError("Cannot handle the case both true and false branchs only \
                             contain const nodes for now.")
        if branch_type_1 == branch_type_2:
            raise ValueError("true graph and false graph are intersected")
        if branch_type_1 == BranchType.TRUE or branch_type_2 == BranchType.FALSE:
            true_branch_nodes = branch_nodes_1
            true_output = merge_input_1
            false_branch_nodes = branch_nodes_2
            false_output = merge_input_2
        else:
            true_branch_nodes = branch_nodes_2
            true_output = merge_input_2
            false_branch_nodes = branch_nodes_1
            false_output = merge_input_1
        return true_branch_nodes, true_output, false_branch_nodes, false_output, switchs

    def _branch_type(self, branch_output, nodes):
        """Infer the branch type (true, false or unknown)"""
        branch = BranchType.UNKNOWN
        # the branch is empty
        if not nodes:
            input_node = self.g.get_node_by_output(branch_output)
            if self._is_switch(input_node):
                if branch_output == input_node.output[0]:
                    branch = BranchType.FALSE
                else:
                    branch = BranchType.TRUE
            return branch
        for node in nodes:
            for inp in node.input:
                input_node = self.g.get_node_by_output(inp)
                if self._is_switch(input_node):
                    if inp == input_node.output[0]:
                        if branch == BranchType.TRUE:
                            raise ValueError("true and false graph intersect at {}".format(node.name))
                        branch = BranchType.FALSE
                    else:
                        if branch == BranchType.FALSE:
                            raise ValueError("true and false graph intersect at {}".format(node.name))
                        branch = BranchType.TRUE
        if branch == BranchType.UNKNOWN:
            logger.debug(
                "branch only contains const node: [%s]",
                ",".join(n.name for n in nodes)
            )
        return branch


def rewrite_cond(g, ops):
    return CondRewriter(g).rewrite()
