# SPDX-License-Identifier: Apache-2.0


"""const fold Optimizer.
   if op's inputs are all const then do op computation when building the graph to improve performance
   for example, input of transpose node is const then we can do transpose statically instead of at runtime
"""

import numpy as np
from .. import utils
from .optimizer_base import GraphOptimizerBase

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring

# key is op_type, value is the function to compute outputs
# the schema of function is: inputs are(node, graph), output is a list of constant values.
_func_map = {}


def _register_func(op_type):
    def _internal_fun(func):
        _func_map[op_type] = func
        return func

    return _internal_fun


class ConstFoldOptimizer(GraphOptimizerBase):

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(ConstFoldOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, graph):
        graph_changed = True
        while graph_changed:
            graph_changed = False
            ops = graph.get_nodes()
            for op in ops:
                if self._should_skip(op):
                    continue
                if self._fold_node(op, graph):
                    graph_changed = True
                    self.graph_been_opt = True
        return graph

    @staticmethod
    def _should_skip(node):
        # only support onnx official op for now, op in other domain is not supported for now
        if not utils.is_onnx_domain(node.domain):
            return True

        if node.is_const() or node.is_graph_input():
            return True

        skip_type = ["Identity", "DequantizeLinear"]
        if node.type in skip_type:
            return True

        return False

    def _fold_node(self, node, graph):
        """ if node's input are all const and it's not graph's output then it can be fold.
            if node can be fold True will be return indicating that graph is changed
        """
        if self._all_inputs_are_const(node.inputs) and not self._is_graph_output(node, graph):
            process_func = _func_map.get(node.type, None)
            if process_func:
                const_outputs = process_func(node, graph)
                self._replace_node_with_const(node, graph, const_outputs)
                return True
            self.logger.debug("need to add function to fold op %s whose op_type is %s", node.name, node.type)
        return False

    @staticmethod
    def compute_const_folding(node, graph):
        return _func_map[node.type](node, graph)

    @staticmethod
    def _all_inputs_are_const(nodes):
        return all(node.is_const() for node in nodes if node)

    @staticmethod
    def _is_graph_output(node, graph):
        node_out_set = set(node.output)
        graph_out_set = set(graph.outputs)
        return node_out_set.intersection(graph_out_set)

    @staticmethod
    def _replace_node_with_const(node, graph, vals):
        utils.make_sure(len(node.output) == len(vals), "length of node outputs and const vals should be same")
        for old_input, val in zip(node.output, vals):
            const_node = graph.make_const(utils.make_name("const_fold_opt"), val)
            graph.set_dtype(const_node.output[0], utils.map_numpy_to_onnx_dtype(val.dtype))
            graph.set_shape(const_node.output[0], val.shape)
            graph.replace_all_inputs(old_input, const_node.output[0])  # ops=graph.get_nodes()
        graph.remove_node(node.name)

    @staticmethod
    @_register_func("Cast")
    def _fold_cast(node, graph):
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[node.get_attr("to").i]
        const_val_after_cast = const_val.astype(np_dtype)
        return [const_val_after_cast]

    @staticmethod
    @_register_func("Transpose")
    def _fold_transpose(node, graph) -> list:
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        perm_attr = node.get_attr("perm")
        perm = perm_attr.ints if perm_attr else None
        const_val_after_trans = const_val.transpose(perm)
        return [const_val_after_trans]

    @staticmethod
    @_register_func("Reshape")
    def _fold_reshape(node, graph):
        const_val_data = node.inputs[0].get_tensor_value(as_list=False)
        const_val_shape = node.inputs[1].get_tensor_value(as_list=True)
        data_shape = const_val_data.shape
        for i, dim in enumerate(const_val_shape):
            if dim == 0:
                # In ORT a dim of 0 means the shape stays the same.
                const_val_shape[i] = data_shape[i]
        const_val_after_trans = const_val_data.reshape(const_val_shape)
        return [const_val_after_trans]

    @staticmethod
    @_register_func("Concat")
    def _fold_concat(node, graph):
        axis = node.get_attr_value('axis')
        res = np.concatenate([inp.get_tensor_value(as_list=False) for inp in node.inputs], axis)
        return [res]

    @staticmethod
    @_register_func("Unsqueeze")
    def _fold_unsqueeze(node, graph):
        """
        numpy expand_dims only supports to unsqueeze one dim one time, so reshape is used to simplify the logic
        """
        const_val = node.inputs[0].get_tensor_value(as_list=False)
        if graph.opset >= 13:
            axes = node.inputs[1].get_tensor_value(as_list=True)
        else:
            axes = list(node.get_attr("axes").ints)
        shape_in = const_val.shape
        dims_out = len(shape_in) + len(axes)
        axes = [i if i >= 0 else i + dims_out for i in axes]
        # calculate the shape of output accroding to onnx Unsqueeze's spec
        # https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze
        shape_in = iter(shape_in)
        shape_out = [None] * dims_out
        for ind in axes:
            shape_out[ind] = 1
        for ind, val in enumerate(shape_out):
            if val is None:
                shape_out[ind] = next(shape_in)

        const_val_after_unsqueeze = const_val.reshape(shape_out)
        return [const_val_after_unsqueeze]

    @staticmethod
    @_register_func("Mul")
    def _fold_mul(node, graph):
        const_val1 = node.inputs[0].get_tensor_value(as_list=False)
        const_val2 = node.inputs[1].get_tensor_value(as_list=False)
        const_val_after_nul = np.multiply(const_val1, const_val2)
        return [const_val_after_nul]

    @staticmethod
    @_register_func("Add")
    def _fold_add(node, graph):
        const_val1 = node.inputs[0].get_tensor_value(as_list=False)
        const_val2 = node.inputs[1].get_tensor_value(as_list=False)
        const_val_after_add = np.add(const_val1, const_val2)
        return [const_val_after_add]

    @staticmethod
    @_register_func("Sub")
    def _fold_sub(node, graph):
        const_val1 = node.inputs[0].get_tensor_value(as_list=False)
        const_val2 = node.inputs[1].get_tensor_value(as_list=False)
        const_val_after_sub = np.subtract(const_val1, const_val2)
        return [const_val_after_sub]

    @staticmethod
    @_register_func("Split")
    def _fold_split(node, graph):
        data = node.inputs[0].get_tensor_value(as_list=False)
        axis = node.get_attr_value('axis', 0)
        if len(node.output) == 1:
            return [data]
        split = node.get_attr_value('split')
        if len(node.input) > 1:
            split = node.inputs[1].get_tensor_value(as_list=False)
        if split is not None:
            indices_or_sections = np.cumsum(split[:-1])
        else:
            indices_or_sections = len(node.output)
        return np.split(data, indices_or_sections, axis)
