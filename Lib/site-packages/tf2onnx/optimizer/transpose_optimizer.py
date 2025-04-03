# SPDX-License-Identifier: Apache-2.0


"""Transpose Optimizer."""

from collections import defaultdict

import numpy as np
import onnx
from tf2onnx.constants import NCHW_TO_NHWC, NHWC_TO_NCHW, NCDHW_TO_NDHWC, NDHWC_TO_NCDHW, TARGET_CHANNELS_LAST
from .. import utils
from .optimizer_base import GraphOptimizerBase


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,abstract-method
# FIXME:
# pylint: disable=unused-variable

def is_transpose(transpose_node):
    perm_attr = transpose_node.get_attr('perm')
    return transpose_node.type == "Transpose" and perm_attr


def is_tranpose_of_type(node, perm):
    perm_attr = node.get_attr('perm')
    return node.type == "Transpose" and  perm_attr and perm_attr.ints == perm


def is_useless_transpose(transpose_node):
    perm_attr = transpose_node.get_attr('perm')
    return transpose_node.type == "Transpose" and perm_attr and perm_attr.ints == list(range(len(perm_attr.ints)))


def get_transpose_rank(trans):
    return len(trans.get_attr('perm').ints)


def invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


class TransposeOptimizer(GraphOptimizerBase):
    """Transpose Optimizer."""

    def __init__(self):
        super(TransposeOptimizer, self).__init__()

        self._handler_map = {}
        self._force_stop = {}

        self._initialize_handlers()
        self._g = None
        self._output_names = None

    @property
    def nodes(self):
        return self._g.get_nodes()

    def pre_optimize_action(self):
        # make Reshape into a const, which then can be fused into Conv's weight for mobilenet_v1_75_192
        self._output_names = [self._g.get_node_by_output(out).name for out in self._g.outputs]
        ops = self.nodes
        constable_reshape_ops = [n for n in ops
                                 if (n.type == "Reshape"
                                     and n.inputs[0].is_const()
                                     and n.inputs[1].is_const())]
        for reshape_op in constable_reshape_ops:
            target_t = reshape_op.inputs[0].get_tensor_value(as_list=False)
            target_shape = reshape_op.inputs[1].get_tensor_value(as_list=True)
            for i, dim in enumerate(target_shape):
                if dim == 0:
                    # In ORT a dim of 0 means the shape stays the same.
                    target_shape[i] = target_t.shape[i]
            new_data = np.reshape(target_t, target_shape)
            const_name = reshape_op.output[0]
            self._g.remove_node(reshape_op.name)
            self._g.make_const(const_name, new_data)

            # point all children nodes inputs to the new node
            for output_name in reshape_op.output:
                for child in ops:
                    for i, name in enumerate(child.input):
                        if name == output_name:
                            child.input[i] = const_name

            self._g.topological_sort(self._g.get_nodes())

    def post_optimize_action(self):
        def _calculate_new_shape(graph, op):
            input_shape = graph.get_shape(op.input[0])
            tagged_shape = [d if d == 1 else "var" + str(i) for i, d in enumerate(input_shape)]
            trim_shape = [d for d in tagged_shape if d != 1]

            perm = op.get_attr_value("perm")
            perm_shape = [tagged_shape[p] for p in perm]
            trim_perm_shape = [d for d in perm_shape if d != 1]

            if trim_perm_shape != trim_shape:
                return None

            if input_shape.count(-1) <= 1:
                new_shape = [input_shape[p] for p in perm]
                return graph.make_const(utils.make_name("new_shape"), np.array(new_shape, dtype=np.int64)).output[0]

            # reshape requires the output shape can only contain one -1, if not some extra op needed.
            input_shape = graph.make_node("Shape", [op.input[0]]).output[0]
            indice = graph.make_const(utils.make_name("indice"), np.array(perm, np.int64)).output[0]

            return graph.make_node("Gather", [input_shape, indice]).output[0]

        nodes = self.nodes
        # if channel==1 or height==width==1, replace transpose with reshape
        # replacing trans with reshape is because transpose will copy data even if this transpose doesn't nothing
        need_sort = False
        for op in nodes:
            if op.type == "Transpose" and "perm" in op.attr:
                input_shape = self._g.get_shape(op.input[0])
                if not input_shape:
                    continue
                new_shape = _calculate_new_shape(self._g, op)
                if new_shape is not None:
                    # replace transpose with reshape
                    shapes = op.output_shapes
                    dtypes = op.output_dtypes
                    self._g.remove_node(op.name)
                    self._g.make_node("Reshape", [op.input[0], new_shape], name=op.name, outputs=op.output,
                                      shapes=shapes, dtypes=dtypes)
                    need_sort = True
        if need_sort:
            self._g.topological_sort(self._g.get_nodes())

    def merge_duplicated_transposes(self):
        # strategy used in previous procedure is to move transpose nodes down if possible,
        # and it means that when a node has n outputs then n transpose will be generated,
        # so we should merge them back to one if they can't be eliminated in previous procedure.
        graph = self._g
        input_transposes_map = defaultdict(list)
        for node in graph.get_nodes():
            if node.type == "Transpose" and node.get_attr("perm"):
                key = (node.input[0], str(node.get_attr("perm").ints))
                input_transposes_map[key].append(node)

        for transposes in input_transposes_map.values():
            # merge transpose nodes into one: make nodes use the output of the first transpose node
            transpose_out = transposes[0].output[0]
            for node in transposes[1:]:
                old_transpose_out = node.output[0]
                graph.replace_all_inputs(old_transpose_out, transpose_out)  # ops=graph.get_nodes()

        # dangling transpose nodes can be deleted
        graph.delete_unused_nodes(graph.outputs)

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        self._g = graph
        self.pre_optimize_action()
        no_action = False
        iteration_cnt = 0
        while not no_action:
            no_action = True
            nodes = self.nodes
            self._force_stop = {}
            for n in nodes:
                if is_transpose(n):
                    if self._handle_nhwc_tranpose(n):
                        no_action = False
                        self.graph_been_opt = True
                        iteration_cnt += 1
                        # need break, because handler may change nodes set, making the n stale object
                        # referencing already deleted elements
                        break

                # Make sure node wasn't already deleted in _handle_nhwc_tranpose
                if graph.get_node_by_name(n.name) is not None and is_useless_transpose(n):
                    no_action = False
                    iteration_cnt += 1
                    self._remove_useless_tranpose(n)
                    break
            # for debugging purpose
            if "stop" in self._force_stop and self._force_stop["stop"] == 1:
                break

        self.logger.debug("finish after " + str(iteration_cnt) + " iteration(s)")

        self.merge_duplicated_transposes()
        self.post_optimize_action()
        return self._g

    def _initialize_handlers(self):
        self._handler_map = {
            "Abs": self._simple_through_handler,
            "Add": self._add_handler,
            "ArgMax": self._arg_min_max_handler,
            "ArgMin": self._arg_min_max_handler,
            "Cast": self._simple_through_handler,
            "Clip": self._simple_through_handler,
            "Concat": self._concat_handler,
            "Elu": self._simple_through_handler,
            "Exp": self._simple_through_handler,
            "Identity": self._identity_handler,
            "LeakyRelu": self._simple_through_handler,
            "Log": self._simple_through_handler,
            "LogSoftmax": self._softmax_handler,
            "Max": self._maxmin_handler,
            "Min": self._maxmin_handler,
            "Mul": self._mul_handler,
            "Neg": self._simple_through_handler,
            "Pad": self._pad_handler,
            "PRelu": self._prelu_handler,
            "Reciprocal": self._simple_through_handler,
            "ReduceLogSum": self._reduce_handler,
            "ReduceLogSumExp": self._reduce_handler,
            "ReduceMax": self._reduce_handler,
            "ReduceMean": self._reduce_handler,
            "ReduceMin": self._reduce_handler,
            "ReduceProd": self._reduce_handler,
            "ReduceSum": self._reducesum_handler,
            "ReduceSumSquare": self._reduce_handler,
            "Relu": self._simple_through_handler,
            "Shape": self._shape_handler,
            "Sigmoid": self._simple_through_handler,
            "Softmax": self._softmax_handler,
            "Sum": self._sum_handler,
            "Slice": self._slice_handler,
            "Split": self._split_handler,
            "Softplus": self._simple_through_handler,
            "Sqrt": self._simple_through_handler,
            "Squeeze": self._squeeze_handler,
            "Sub": self._sub_handler,
            "Unsqueeze": self._unsqueeze_handler,
            "Tanh": self._simple_through_handler,
            "Tile": self._tile_handler,
            "Transpose": self._transpose_handler,
            "DequantizeLinear": self._quantize_handler,
            "QuantizeLinear": self._quantize_handler,
        }

    def _handle_node_having_branches(self, trans, node):
        if not self._should_push_transpose(trans, node) or len(node.output) != 1:
            return False
        # create transpose pairs if some input are not.
        if not self._create_transpose_pairs_before_node(trans, node):
            return False
        # make sure node's all input transpose all have only 1 consumer node,
        # otherwise, it would impact their other output nodes
        if self._nodes_has_single_consumer_node(node.inputs):
            self._create_transpose_pairs_after_node(trans, node)
            input_transposes = set(node.inputs)
            for n in input_transposes:
                n_input = n.input[0]
                utils.make_sure(len(n.output) == 1, "only expect single output")
                self._g.replace_all_inputs(n.output[0], n_input)  # ops=self._g.get_nodes()
                self._g.remove_node(n.name)

            utils.make_sure(len(node.output) == 1, "only expect single output")
            # currently we assume node only has 1 output, for cases where it is more than 1 for example Split
            # we need consider the fact that Split's multiple output will not always has data in NCHW/NHWC,
            # it might be a different shape.
            output_transposes = self._g.find_output_consumers(node.output[0])
            for n in output_transposes:
                n_input = n.input[0]
                utils.make_sure(len(n.output) == 1, "only expect single output")
                self._g.replace_all_inputs(n.output[0], n_input)  # ops=self._g.get_nodes()
                self._g.remove_node(n.name)

            shape = self._g.get_shape(node.output[0])
            if shape:
                # only nhwc transpose can reach here
                perm_inv = invert_perm(trans.get_attr_value("perm"))
                new_shape = [shape[i] for i in perm_inv]
                self._g.set_shape(node.output[0], new_shape)
            return True

        self.logger.debug("input transpose does not have single consumer, skipping...")
        return False

    # get the input index of transpose op in node's inputs.
    def _get_input_index_for_trans(self, node, trans):
        input_index = 0
        for i in node.input:
            if i == trans.output[0]:
                break
            input_index += 1
        return input_index

    # the assumption is: both node and trans have only 1 output
    def _switch_transpose_and_node(self, node, trans, update_shape=True):
        if not self._nodes_has_single_consumer_node([trans]):
            return False

        input_index = self._get_input_index_for_trans(node, trans)

        self._g.replace_all_inputs(node.output[0], trans.output[0])  # ops=self._g.get_nodes()
        self._g.replace_input(node, node.input[input_index], trans.input[0], input_index)
        self._g.replace_input(trans, trans.input[0], node.output[0], 0)

        # need to transpose node shape in backward direction as well after switch
        # otherwise, reshape added in post_optimize_action may not work correctly
        shape = self._g.get_shape(node.output[0])
        if update_shape and shape:
            # only nhwc transpose can reach here
            perm_inv = invert_perm(trans.get_attr_value("perm"))
            new_shape = [shape[i] for i in perm_inv]
            self._g.set_shape(node.output[0], new_shape)
            self._g.set_shape(trans.output[0], shape)
        return True

    # if return value is True, then it means Transpose is handled as designed
    # otherwise, it means that we skip handling since it is not in our support set
    def _handle_nhwc_tranpose(self, trans):
        if trans.output[0] in self._g.outputs:
            self.logger.debug("%s connects to graph outputs, skip", trans.output[0])
            return False
        out_nodes = self._g.find_output_consumers(trans.output[0])
        if len(out_nodes) == 1:
            p = out_nodes[0]
            if p.name in self._output_names:
                self.logger.debug("cannot move transpose down since it met output node %s", p.name)
                return False

            if p.type in self._handler_map:
                op_handler = self._handler_map[p.type]
                return op_handler(trans, p)
            return False
        if out_nodes and trans.get_attr_value("perm") in [NCHW_TO_NHWC, NCDHW_TO_NDHWC]:
            # Move transpose into branches to let Transposes can be "handled" in each branch.
            # This will add more transpose ops, so only do this if further optimization is likely (check perm).
            for n in out_nodes:
                branch_trans = n.graph.make_node("Transpose", [trans.input[0]], attr=trans.get_onnx_attrs())
                n.graph.replace_input(n, trans.output[0], branch_trans.output[0])
            self._g.remove_node(trans.name)
        return False

    def _remove_useless_tranpose(self, trans):
        self._g.replace_all_inputs(trans.output[0], trans.input[0])  # ops=self._g.get_nodes()
        self._g.remove_node(trans.name)

    def _nodes_has_single_consumer_node(self, nodes):
        for n in nodes:
            for output in n.output:
                cnt = len(set(self._g.find_output_consumers(output)))
                if cnt != 1:
                    return False
        return True

    def _cost_to_transpose(self, node, inp_id):
        if node.type in ["Const", "Transpose"]:
            # Transposes can be combined/folded, so there is no additional cost
            return 0
        prod = 1
        shape = self._g.get_shape(inp_id)
        if shape is None:
            return 500
        for d in shape:
            if d == -1:
                # Assume unknown dims are approx. 20
                prod *= 20
            else:
                prod *= d
        return prod

    def _should_push_transpose(self, trans, node):
        perm = trans.get_attr_value("perm")
        optimization_gains = 0
        removed_nchws = 0
        perm_to_push_down = [NCHW_TO_NHWC, NCDHW_TO_NDHWC]
        perm_to_push_up = [NHWC_TO_NCHW, NDHWC_TO_NCDHW]
        if self._g.is_target(TARGET_CHANNELS_LAST):
            perm_to_push_down, perm_to_push_up = perm_to_push_up, perm_to_push_down

        for n, inp_id in zip(node.inputs, node.input):
            if is_tranpose_of_type(n, perm):
                optimization_gains += self._cost_to_transpose(n.inputs[0], n.input[0])
                if perm in perm_to_push_down:
                    removed_nchws += 1
            else:
                optimization_gains -= self._cost_to_transpose(n, inp_id)
                if perm in perm_to_push_up:
                    removed_nchws -= 1
        if removed_nchws != 0:
            # Always push nchw transposes if possible
            return removed_nchws > 0
        return optimization_gains > 0

    def _get_non_nchw_transpose_output_nodes(self, trans, node):
        # we just support node having 1 output, we need consider cases where node has more than 1 outputs
        assert len(node.output) == 1
        perm = trans.get_attr_value("perm")
        perm_inv = invert_perm(perm)
        non_nchw_tranpose_nodes = []
        consumers = self._g.find_output_consumers(node.output[0])
        for o in consumers:
            if not is_tranpose_of_type(o, perm_inv) and o not in non_nchw_tranpose_nodes:
                non_nchw_tranpose_nodes.append(o)
        return non_nchw_tranpose_nodes

    def _create_transpose_pairs_after_node(self, trans, node):
        assert len(node.output) == 1  # just support node who has 1 output
        non_nchw_trans_consumers = self._get_non_nchw_transpose_output_nodes(trans, node)
        # add Transpose(0, 3, 1, 2) and Transpose(0, 2, 3, 1) before each non_nchw_trans_consumers
        for consumer in non_nchw_trans_consumers:
            perm = trans.get_attr_value("perm")
            perm_inv = invert_perm(perm)
            nchw_node = self._g.make_node("Transpose", [node.output[0]], attr={"perm": perm_inv})
            nhwc_node = self._g.make_node("Transpose", [nchw_node.output[0]], attr={"perm": perm})
            self._g.replace_input(consumer, node.output[0], nhwc_node.output[0])

    def _create_transpose_pairs_before_node(self, trans, node):
        perm = trans.get_attr_value("perm")
        perm_inv = invert_perm(perm)
        trans_rank = len(perm)
        def shape_after_expand(ori_shape):
            # according to broadcasting rule to expand shape to 4D while not tile the tensor here
            # still count on the broadcasting op to tile the tensor
            if ori_shape.count(-1) >= 2:
                self.logger.warning("%s shape can contain one -1 at most, otherwise reshape op can't work", node.name)
                return None
            ori_rank = len(ori_shape)
            new_shape = [1] * (trans_rank - ori_rank) + ori_shape
            return new_shape

        non_nhwc_trans_inputs = []
        for input_id, n in zip(node.input, node.inputs):
            if not is_tranpose_of_type(n, perm):
                # check in case node has two inputs coming from a same node output.
                if [input_id, n] not in non_nhwc_trans_inputs:
                    non_nhwc_trans_inputs.append([input_id, n])

        # add Transpose NHWC_TO_NCHW and Transpose NCHW_TO_NHWC before each non_nhwc_trans_consumers
        shape_unknow = [input_id for input_id, _ in non_nhwc_trans_inputs if self._g.get_shape(input_id) is None]
        if shape_unknow:
            if self._g.opset <= 9:
                msg = "%s 's shape is unknown, ConstantOfShape will be used which exists in version 9 or higher" \
                      "while graph's opset version is %s" % (shape_unknow, self._g.opset)
                self.logger.warning(msg)
                return False

        for input_id, n in non_nhwc_trans_inputs:
            shape = self._g.get_shape(input_id)
            # if rank of n is not transpose rank, then we need to insert a reshape op before inserting a transpose
            # for example shape of n is [x, y], then output shape of reshape will be [1, 1, x, y] or [1, 1, 1, x, y]
            if shape is None:
                const_4 = self._g.make_const(utils.make_name("const_4"), np.array([trans_rank], np.int64)).output[0]
                tensor_1 = onnx.helper.make_tensor("value", onnx.TensorProto.INT64, [1], [1])
                shape_node = self._g.make_node("Shape", [input_id]).output[0]
                rank_node = self._g.make_node("Shape", [shape_node]).output[0]
                expand_rank = self._g.make_node("Sub", [const_4, rank_node]).output[0]
                array_fill_1 = self._g.make_node("ConstantOfShape", [expand_rank], attr={"value": tensor_1}).output[0]
                new_shape = self._g.make_node("Concat", [array_fill_1, shape_node], attr={"axis": 0}).output[0]
                reshape = self._g.make_node("Reshape", [input_id, new_shape]).output[0]
                input_of_new_trans = reshape
            elif len(shape) == trans_rank:
                input_of_new_trans = input_id
            else:
                shape = shape_after_expand(shape)
                if shape is None:
                    return False
                const = self._g.make_const(utils.make_name("reshape_shape"), np.array(shape, np.int64)).output[0]
                reshape = self._g.make_node("Reshape", [input_id, const]).output[0]
                input_of_new_trans = reshape

            nchw_node = self._g.make_node("Transpose", [input_of_new_trans], attr={"perm": perm_inv})
            nhwc_node = self._g.make_node("Transpose", [nchw_node.output[0]], attr={"perm": perm})
            self._g.replace_input(node, input_id, nhwc_node.output[0])
        return True

    def _add_handler(self, trans, node):
        if node.inputs[1].is_const():
            t_p = trans.inputs[0]
            if t_p.type in ("Conv", "ConvTranspose") and len(t_p.input) == 2:
                # if Conv or ConvTranspose's bias input is not set, then we set, otherwise, we don't set
                # todo: maybe we can add already set bias with the input??? try later

                if not self._nodes_has_single_consumer_node([t_p]):
                    self.logger.debug("Conv does not have single consumer, can not merge Conv and Add")
                    return self._handle_node_having_branches(trans, node)

                if not self._nodes_has_single_consumer_node([trans]):
                    self.logger.debug("input transpose does not have single consumer, skipping...")
                    return False

                target_node = node.inputs[1]
                numpy_val = target_node.get_tensor_value(as_list=False)
                # Optional 1D bias to be added to the convolution, has size of M
                if len(numpy_val.shape) - numpy_val.shape.count(1) > 1:
                    self.logger.debug("Bias is not 1D, can not merge Conv and Add")
                    return self._handle_node_having_branches(trans, node)

                bias_size = max(numpy_val.shape)
                size_m = t_p.inputs[1].output_shapes[0][0]
                if bias_size != size_m:
                    self.logger.debug("Bias size is not M, can not merge Conv and Add")
                    return self._handle_node_having_branches(trans, node)

                target_val = numpy_val.reshape(bias_size)
                target_node.set_tensor_value(target_val)

                conv_inputs = [t_p.input[0], t_p.input[1], node.input[1]]
                conv_node = self._g.make_node(t_p.type, conv_inputs, attr=t_p.get_onnx_attrs())
                self._g.replace_input(trans, trans.input[0], utils.port_name(conv_node.name), 0)
                self._g.replace_all_inputs(node.output[0], trans.output[0])  # ops=self._g.get_nodes()
                self._g.remove_node(t_p.name)
                self._g.remove_node(node.name)
                return True
        return self._handle_node_having_branches(trans, node)

    def _output_node_has_single_consumer_node(self, node):
        output_node = self._g.get_node_by_name(node.output[0])
        return output_node and output_node.output and self._nodes_has_single_consumer_node([output_node])

    def _transpose_handler(self, trans, node):
        perm = trans.get_attr_value("perm")
        perm_inv = invert_perm(perm)
        if is_tranpose_of_type(node, perm_inv) and self._output_node_has_single_consumer_node(node):
            for g in {self._g, node.graph}:
                g.replace_all_inputs(node.output[0], trans.input[0])  # ops=g.get_nodes()

            shape = node.graph.get_shape(node.output[0])
            dtype = node.graph.get_dtype(node.output[0])
            if node.output[0] in node.graph.outputs:
                node.graph.make_node("Identity", [trans.input[0]],
                                     outputs=node.output, shapes=[shape], dtypes=[dtype])
            self._g.remove_node(trans.name)
            node.graph.remove_node(node.name)
            return True
        return False

    def _maxmin_handler(self, trans, node):
        return self._handle_node_having_branches(trans, node)

    def _mul_handler(self, trans, node):
        multiplier_input_id = None
        multiplier_input_node = None
        multiplier_input_idx = None
        for idx, (input_id, input_node) in enumerate(zip(node.input, node.inputs)):
            if input_id != trans.output[0]:
                multiplier_input_id = input_id
                multiplier_input_node = input_node
                multiplier_input_idx = idx

        # node's inputs may come from one same node. if so the multiplier_input_node may be none
        if multiplier_input_node is None:
            if not self._nodes_has_single_consumer_node([trans]):
                return False
            self._g.replace_all_inputs(node.output[0], trans.output[0])
            self._g.replace_input(node, node.input[0], trans.input[0], 0)
            self._g.replace_input(node, node.input[1], trans.input[0], 1)
            self._g.replace_input(trans, trans.input[0], node.output[0], 0)
            return True

        # convert  mul(trans(x), trans(y)) ->  trans(mul(x, y))
        if is_tranpose_of_type(multiplier_input_node, trans.get_attr_value("perm")):
            if not self._nodes_has_single_consumer_node([multiplier_input_node]):
                return False
            input_index = self._get_input_index_for_trans(node, multiplier_input_node)
            if not self._switch_transpose_and_node(node, trans):
                return False

            self._g.replace_input(node, node.input[input_index], multiplier_input_node.input[0], input_index)
            self._g.remove_node(multiplier_input_node.name)
            return True

        # handle const multipliers
        if not multiplier_input_node.is_const():
            return False
        multiplier = multiplier_input_node.get_tensor_value(as_list=False)

        # todo: apply this block if we have model case multiplier_input_id==0, and verify that.
        if multiplier_input_id == node.input[1]:
            t_p = trans.inputs[0]
            trans_rank = get_transpose_rank(trans)
            # make sure conv don't have bias set
            can_opt = t_p.type == "Conv" and t_p.inputs[1].is_const() and len(t_p.input) == 2 and trans_rank == 4
            can_opt = can_opt and self._nodes_has_single_consumer_node([t_p])
            # make sure multiplier with shape (N,) or (1, N) or (1, 1, N) ....
            can_opt = can_opt and trans.get_attr_value("perm") == NCHW_TO_NHWC \
                and all(shape == 1 for shape in multiplier.shape[:-1])
            if can_opt:
                conv = t_p
                numpy_val = conv.inputs[1].get_tensor_value(as_list=False)
                transposed_val = np.transpose(numpy_val, (2, 3, 1, 0))
                mul_val = multiplier
                result = np.multiply(transposed_val, mul_val)
                conv.inputs[1].set_tensor_value(np.transpose(result, (3, 2, 0, 1)))

                self._g.replace_all_inputs(node.output[0], trans.output[0])  # ops=self._g.get_nodes()
                self._g.remove_node(node.name)
                return True

        # if the shape is (), we just move transpose after the mul
        if not multiplier.shape:
            return self._switch_transpose_and_node(node, trans)

        # if multiplier is 1-D
        if len(multiplier.shape) == 1 and multiplier.shape[0] == 1:
            # shape is (1)
            return self._switch_transpose_and_node(node, trans)

        # if multiplier has shape (N,) or (1, N) or (1, 1, N) ....
        if np.prod(multiplier.shape) == multiplier.shape[-1]:
            if not self._nodes_has_single_consumer_node([multiplier_input_node]):
                new_inp = self._g.copy_const(multiplier_input_node)
                self._g.replace_input(node, multiplier_input_id, new_inp.output[0], multiplier_input_idx)
                multiplier_input_node = new_inp
            perm = list(trans.get_attr('perm').ints)
            new_shape = np.ones(len(perm), dtype=np.int32)
            new_shape[perm[-1]] = multiplier.shape[-1]
            multiplier_input_node.set_tensor_value(multiplier.reshape(new_shape))
            return self._switch_transpose_and_node(node, trans)

        return False

    def _sum_handler(self, trans, node):
        inputs = node.inputs
        trans_shape = self._g.get_shape(trans.output[0])
        perm = list(trans.get_attr('perm').ints)
        untrans_idx = [perm.index(i) for i in range(len(perm))]

        # check if sum(trans(x1), trans(x2), const(x3), ...) can be switched
        for n in inputs:
            if n.type not in ["Transpose", "Const"]:
                return False
            if not self._nodes_has_single_consumer_node([n]):
                return False
            if n.is_const():
                # if graph is valid, op shapes should be valid
                # const is special case, in case of broadcasting
                # ensure rank matches
                n_shape = self._g.get_shape(n.output[0])
                if len(n_shape) != len(trans_shape):
                    return False
            else:
                if list(n.get_attr('perm').ints) != perm:
                    return False

        # switch to trans(sum(x1, x2, x3, ...))
        self._g.replace_all_inputs(node.output[0], trans.output[0])  # ops=self._g.get_nodes()
        new_input = [n.output[0] if n.is_const() else n.input[0] for n in inputs]
        self._g.replace_inputs(node, new_input)
        self._g.replace_input(trans, trans.input[0], node.output[0], 0)

        # adjust shape if present
        shape = self._g.get_shape(node.output[0])
        if shape:
            self._g.set_shape(node.output[0], [shape[i] for i in untrans_idx])

        # update constants, remove dangling transposes
        for n in inputs:
            if n.is_const():
                val = n.get_tensor_value(as_list=False)
                new_val = np.transpose(val, untrans_idx)
                n.set_tensor_value(new_val)
            elif n.name != trans.name:
                self._g.remove_node(n.name)
        return True

    def _identity_handler(self, trans, node):
        if node.output[0] in node.graph.outputs:
            return False
        for g in {self._g, node.graph}:
            g.replace_all_inputs(node.output[0], trans.output[0])  # ops=g.get_nodes()
        node.graph.remove_node(node.name)
        return True

    def _concat_handler(self, trans, node):
        if self._handle_node_having_branches(trans, node):
            perm = trans.get_attr_value("perm")
            axis = node.get_attr_value("axis", 0)
            new_axis = perm[axis]
            node.set_attr("axis", new_axis)
            return True
        return False

    def _split_handler(self, trans, node):
        # Todo: need handle cases where Split node has more than 1 outputs.
        split = None
        if self._g.opset >= 13 and len(node.input) > 1 and node.inputs[1].is_const():
            # split is an input not attr since opset 13
            split = node.inputs[1].get_tensor_value(as_list=True)
        if self._handle_node_having_branches(trans, node):
            perm = trans.get_attr_value("perm")
            axis = node.get_attr_value("axis", 0)
            new_axis = perm[axis]
            node.set_attr("axis", new_axis)
            if split:
                new_axes_np = np.array(split, dtype=np.int64)
                new_axes_const = self._g.make_const(utils.make_name(node.inputs[1].name), new_axes_np)
                self._g.replace_inputs(node, [node.input[0], new_axes_const.output[0]])
            return True
        return False

    def _unsqueeze_handler(self, trans, node):
        trans_rank = get_transpose_rank(trans)
        perm = trans.get_attr_value("perm")
        axes = None
        if node.get_attr("axes"):
            axes = node.get_attr("axes").ints
        if len(node.input) > 1 and node.inputs[1].is_const():
            axes = node.inputs[1].get_tensor_value(as_list=True)
        if axes is None:
            return False

        new_rank = trans_rank + len(axes)
        axes = sorted([a % new_rank for a in axes])
        # We have a choice of where to put the new axes for unsqueeze after we push the transpose. We will try to keep
        # them next to the axis they will be next to after transpose ex: a1bc -> ac1b not 1abc -> ac1b
        partner_axes = [a - i for i, a in enumerate(axes)]
        pre_perm_axes = [perm[a] if a < len(perm) else len(perm) for a in partner_axes]
        pre_perm_sorted = sorted((a, i) for i, a in enumerate(pre_perm_axes))
        new_axes = [a + pre_perm_sorted.index((a, i)) for i, a in enumerate(pre_perm_axes)]

        shift_map = []
        for i in range(new_rank):
            if i not in new_axes:
                shift_map.append(i)

        new_perm = []
        perm_i = 0
        axes_i = 0
        for i in range(new_rank):
            if i in axes:
                new_perm.append(new_axes[axes_i])
                axes_i += 1
            else:
                new_perm.append(shift_map[perm[perm_i]])
                perm_i += 1

        if not self._switch_transpose_and_node(node, trans, update_shape=False):
            return False

        new_axes_sorted = sorted(new_axes)
        trans.set_attr("perm", new_perm)
        new_perm_inv = invert_perm(new_perm)
        if self._g.opset <= 12:
            node.set_attr("axes", new_axes_sorted)
        else:
            new_axes_np = np.array(new_axes_sorted, dtype=np.int64)
            new_axes_const = self._g.make_const(utils.make_name(node.inputs[1].name), new_axes_np)
            self._g.replace_inputs(node, [node.input[0], new_axes_const.output[0]])

        shape = self._g.get_shape(node.output[0])
        if shape is not None:
            self._g.set_shape(trans.output[0], shape)
            mid_shape = [shape[p] for p in new_perm_inv]
            self._g.set_shape(node.output[0], mid_shape)

        return True

    def _squeeze_handler(self, trans, node):
        trans_rank = get_transpose_rank(trans)
        def _calculate_new_attr(ori_perm, ori_squeeze_axes):
            ori_squeeze_axes = [i if i >= 0 else i + trans_rank for i in ori_squeeze_axes]
            new_squeeze_axes = sorted([ori_perm[i] for i in ori_squeeze_axes])
            # calculate output shape after trans and squeeze
            n = len(ori_perm)
            input_shape = list(range(n))
            shape_after_trans = [input_shape[i] for i in ori_perm]
            output_shape = [shape_after_trans[i] for i in range(n) if i not in ori_squeeze_axes]
            # calculate new_perm
            # after switch, the output shape should be same, using this condition we can figure the new perm
            shape_after_squeeze = [input_shape[i] for i in range(n) if i not in new_squeeze_axes]
            new_perm = [shape_after_squeeze.index(i) for i in output_shape]

            return new_perm, new_squeeze_axes

        if not self._nodes_has_single_consumer_node([trans]):
            return False

        axes = None
        # axes is an input not attr since opset 13
        if node.get_attr("axes"):
            axes = node.get_attr("axes").ints
        if len(node.input) > 1 and node.inputs[1].is_const():
            axes = node.inputs[1].get_tensor_value(as_list=True)

        if axes is not None:
            # switch tran and squeeze
            # 1 switch
            self._g.replace_all_inputs(node.output[0], trans.output[0])  # ops=self._g.get_nodes()
            self._g.replace_input(node, node.input[0], trans.input[0], 0)
            self._g.replace_input(trans, trans.input[0], node.output[0], 0)
            # 2 correct attr of nodes
            squeeze_axes = sorted(axes)
            trans_perm = list(trans.get_attr("perm").ints)
            new_perm, new_squeeze_axes = _calculate_new_attr(ori_perm=trans_perm, ori_squeeze_axes=squeeze_axes)
            trans.set_attr("perm", new_perm)
            if self._g.opset <= 12:
                node.set_attr("axes", new_squeeze_axes)
            else:
                new_axes_np = np.array(new_squeeze_axes, dtype=np.int64)
                new_axes_const = self._g.make_const(utils.make_name(node.inputs[1].name), new_axes_np)
                self._g.replace_inputs(node, [node.input[0], new_axes_const.output[0]])
            # 3 set shape
            squeeze_shape = self._g.get_shape(node.output[0])
            self._g.set_shape(trans.output[0], squeeze_shape)
            input_shape = self._g.get_shape(node.input[0])
            if input_shape is not None:
                new_squeeze_output_shape = [input_shape[i] for i in range(trans_rank) if i not in new_squeeze_axes]
            else:
                new_squeeze_output_shape = [-1] * (trans_rank - len(new_squeeze_axes))
            self._g.set_shape(node.output[0], new_squeeze_output_shape)
            return True
        return False

    def _sub_handler(self, trans, node):
        return self._handle_node_having_branches(trans, node)

    def _pad_handler(self, trans, node):
        trans_rank = get_transpose_rank(trans)
        perm_inv = invert_perm(trans.get_attr_value("perm"))
        # [N-start, H-start, W-start, C-start, N-end, H-end,  W-end, C-end]
        def permute_pads(pads):
            return [pads[i] for i in perm_inv] + [pads[i + trans_rank] for i in perm_inv]

        if self._g.opset < 11:
            pads = node.get_attr('pads').ints  # [x1_begin, x2_begin...x1_end, x2_end,...]
            new_pads = np.array(permute_pads(pads), np.int64)
            node.set_attr("pads", new_pads)
            return self._switch_transpose_and_node(node, trans)

        input1 = node.inputs[1]
        if input1.is_const():
            if not self._nodes_has_single_consumer_node([input1]):
                input1 = self._g.copy_const(input1)
                self._g.replace_input(node, node.input[1], input1.output[0], 1)
            pads = input1.get_tensor_value()
            new_pads = np.array(permute_pads(pads), np.int64)
            input1.set_tensor_value(new_pads)
            return self._switch_transpose_and_node(node, trans)
        # when the second input is not a constant, let's shuffle it with Split followed by Concat
        # there are examples of models, where this non-constant input
        # gets constant folded anyway by a framework.
        split = self._g.make_node("Split", inputs=[node.input[1]], attr={'num_outputs': trans_rank * 2},
                                  output_count=trans_rank * 2)
        pads = split.output
        new_pads = self._g.make_node("Concat", permute_pads(pads), {'axis': 0})
        self._g.replace_input(node, node.input[1], new_pads.output[0], 1)
        return self._switch_transpose_and_node(node, trans)

    def _prelu_handler(self, trans, node):
        return self._handle_node_having_branches(trans, node)

    def _softmax_handler(self, trans, node):
        trans_rank = get_transpose_rank(trans)
        perm = trans.get_attr("perm").ints

        if self._g.opset >= 13:
            # Softmax operates on an arbitrary axis since opset 13
            axis = node.get_attr_value("axis", -1)
            new_axis = perm[axis + trans_rank if axis < 0 else axis]
            if not self._switch_transpose_and_node(node, trans):
                return False
            node.set_attr("axis", new_axis)
            return True

        # For older opsets, the "axis" attribute determines the coercion point for coercing the input tensor to 2D.
        # We can safely switch transpose and node if the permutation does not make any axes cross that boundary.
        coercion_axis = node.get_attr_value("axis", 1)
        for from_axis, to_axis in enumerate(perm):
            if (from_axis < coercion_axis <= to_axis) or (from_axis >= coercion_axis > to_axis):
                return False

        return self._switch_transpose_and_node(node, trans)

    def _arg_min_max_handler(self, trans, node):
        axis = node.get_attr_value("axis", 0)
        node.set_attr("axes", [axis])
        result = self._reduce_base_handler(trans, node)
        new_axis = node.get_attr_value("axes")[0]
        node.set_attr("axis", new_axis)
        del node.attr["axes"]
        return result

    def _reduce_base_handler(self, trans, node):
        keepdims = node.get_attr_value("keepdims", 1)
        trans_rank = get_transpose_rank(trans)
        axes = node.get_attr_value("axes", list(range(trans_rank)))
        perm = trans.get_attr("perm").ints
        axes = [a + trans_rank if a < 0 else a for a in axes]
        new_axes = [perm[a] for a in axes]
        update_shape = keepdims == 1
        shape = self._g.get_shape(node.output[0])
        if not self._switch_transpose_and_node(node, trans, update_shape):
            return False
        node.set_attr("axes", new_axes)
        if keepdims == 0:
            remaining_axes = []
            j = 0
            for i in range(trans_rank):
                if i in new_axes:
                    remaining_axes.append(None)
                else:
                    remaining_axes.append(j)
                    j += 1
            new_perm = [remaining_axes[p] for p in perm if remaining_axes[p] is not None]
            if shape:
                new_shape = [shape[new_perm.index(i)] for i in range(len(new_perm))]
                self._g.set_shape(node.output[0], new_shape)
            trans.set_attr("perm", new_perm)
        return True

    def _reduce_latest_handler(self, trans, node):
        keepdims = node.get_attr("keepdims")
        if keepdims and keepdims.i == 0:
            return False
        if node.inputs[1].is_const():
            axes = node.inputs[1].get_tensor_value()
            perm = trans.get_attr('perm').ints
            axes = [perm[axes[i]] for i in range(len(axes))]
            new_axes = np.array(axes, dtype=np.int64)
            if self._nodes_has_single_consumer_node([node.inputs[1]]):
                node.inputs[1].set_tensor_value(new_axes)
            else:
                new_axes_const = self._g.make_const(
                    utils.make_name(node.inputs[1].name), new_axes
                )
                self._g.replace_input(node, node.input[1], new_axes_const.output[0], 1)
            return self._switch_transpose_and_node(node, trans)
        return False

    def _reduce_handler(self, trans, node):
        if self._g.opset < 18:
            return self._reduce_base_handler(trans, node)
        return self._reduce_latest_handler(trans, node)

    def _tile_handler(self, trans, node):
        if not node.inputs[1].is_const():
            return False
        if not self._switch_transpose_and_node(node, trans):
            return False
        repeats = node.inputs[1].get_tensor_value()
        perm_inv = invert_perm(trans.get_attr_value("perm"))
        repeats_val = [repeats[p] for p in perm_inv]
        new_repeats = np.array(repeats_val, dtype=np.int64)
        if not self._nodes_has_single_consumer_node([node.inputs[1]]):
            new_inp = self._g.copy_const(node.inputs[1])
            self._g.replace_input(node, node.input[1], new_inp.output[0], 1)
        node.inputs[1].set_tensor_value(new_repeats)
        return True

    def _reducesum_handler(self, trans, node):
        if self._g.opset < 13:
            return self._reduce_base_handler(trans, node)
        return self._reduce_latest_handler(trans, node)

    def _slice_handler(self, trans, node):
        axes = None
        if self._g.opset < 10:
            axes_values = node.get_attr("axes")
            if not axes_values:
                return False
            axes = axes_values.ints
            perm = trans.get_attr_value("perm")
            new_axes = [perm[axes[i]] for i in range(len(axes))]
            node.set_attr("axes", new_axes)
            return self._switch_transpose_and_node(node, trans)
        # in opset 10, axes is input instead of an attribute.
        if len(node.inputs) >= 4 and node.inputs[3].is_const():
            axes = node.inputs[3].get_tensor_value(as_list=False)
            dtype = axes.dtype
            axes = axes.tolist()
            perm = trans.get_attr_value("perm")
            axes = [perm[axes[i]] for i in range(len(axes))]
            # axes node might be shared
            new_axes = np.array(axes, dtype=dtype)
            if self._nodes_has_single_consumer_node([node.inputs[3]]):
                node.inputs[3].set_tensor_value(new_axes)
            else:
                new_axes_const = self._g.make_const(
                    utils.make_name(node.inputs[3].name), new_axes
                )
                self._g.replace_input(node, node.input[3], new_axes_const.output[0], 3)
            return self._switch_transpose_and_node(node, trans)
        return False

    def _quantize_handler(self, trans, node):
        # Used for QuantizeLinear and DequantizeLinear
        if node.type == "DequantizeLinear":
            # Only push through if we will be able to push through consumers too.
            cons = self._g.find_output_consumers(node.output[0])
            # If there is a false positive in the handler map, the q_dq and transpose optimizers might fight.
            # Give up after 3 iterations. The q_dq optimizer should win so the dq hugs the op.
            if not all(n.type in self._handler_map for n in cons) or self.opt_iteration >= 3:
                return False
        if not self._switch_transpose_and_node(node, trans):
            return False
        if 'axis' in node.attr:
            perm = trans.get_attr_value("perm")
            axis = node.get_attr_value("axis")
            new_axis = perm[axis]
            node.set_attr("axis", new_axis)
        return True

    def _simple_through_handler(self, trans, node):
        return self._switch_transpose_and_node(node, trans)

    def _shape_handler(self, trans, node):
        # input > trans > shape  can be changed into  input > shape > gather
        if not self._nodes_has_single_consumer_node([trans]):
            return False

        output_shape = self._g.get_shape(node.output[0])
        output_dtype = self._g.get_dtype(node.output[0])
        self._g.remove_node(trans.name)
        self._g.remove_node(node.name)
        shape_node = self._g.make_node("Shape", [trans.input[0]])
        const_node = self._g.make_const(utils.make_name("Const"), np.array(trans.get_attr("perm").ints))
        gather_node = self._g.make_node("Gather", [shape_node.output[0], const_node.output[0]], outputs=node.output)
        self._g.set_shape(gather_node.output[0], output_shape)
        self._g.set_dtype(gather_node.output[0], output_dtype)
        return True
