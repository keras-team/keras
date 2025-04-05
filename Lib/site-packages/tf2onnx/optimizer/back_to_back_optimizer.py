# SPDX-License-Identifier: Apache-2.0


"""Back_To_Back Optimizer.
   Collapse consecutive nodes into 1 node if possible.
"""

import numpy as np
from tf2onnx.utils import ONNX_DTYPE_NAMES  # lgtm[py/unsafe-cyclic-import]
from .optimizer_base import GraphOptimizerBase  # lgtm[py/unsafe-cyclic-import]

# pylint: disable=logging-not-lazy,unused-argument,missing-docstring,unused-variable,arguments-differ

_func_map = {}


def _register_func(op_type):
    if not isinstance(op_type, tuple):
        op_type = (op_type,)
    def _internal_fun(func):
        _func_map[op_type] = func
        return func

    return _internal_fun


class BackToBackOptimizer(GraphOptimizerBase):
    """Remove back-to-back nodes e.g. 'Cast'
    """

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(BackToBackOptimizer, self).__init__()

    def _optimize(self, graph):
        return self._apply_optimization(graph, self._optimize_at_current_graph_level)

    def _optimize_at_current_graph_level(self, g):
        for optype, handler in _func_map.items():
            # candidate nodes for removal/optimization
            nodes = [n for n in g.get_nodes() if n.type in optype]

            # topological sort of candidates
            # simplifying assumption for back-to-back-optimizer is
            # the op_types have 1 input, 1 output, but multiple consumers.
            # if optype contains 2 elements, the second element should not be considered as a consumer.
            has_dependencies = set()
            consumer_node_ids = {n.output[0]: [] for n in nodes if len(optype) < 2 or n.type == optype[0]}
            for n in nodes:
                if n.input[0] in consumer_node_ids:
                    consumer_node_ids[n.input[0]].extend([n])
                    has_dependencies.add(n.output[0])

            # q = starting nodes with no dependencies
            q = list(set(consumer_node_ids.keys()) - has_dependencies)
            while q:
                nodeid = q.pop(0)
                node = g.get_node_by_output(nodeid, False)
                consumer_nodes = consumer_node_ids[nodeid]

                if len(consumer_nodes) > 0:
                    all_consumers = g.find_output_consumers(node.output[0])
                    if len(all_consumers) != len(consumer_nodes):
                        # if first node is used elsewhere, skip
                        continue
                    if set(node.output) & set(g.outputs):
                        # if this node is part of graph outputs, skip
                        continue
                    q2 = handler(g, node, consumer_nodes)
                    # add more nodes which can now be processed
                    q.extend(q2)
        return g

    @staticmethod
    @_register_func("Cast")
    def _optimize_cast(g, node, consumer_nodes):
        """remove long chains of cast ops"""
        q2 = []
        type1 = node.get_attr('to').i
        type1_name = ONNX_DTYPE_NAMES[type1] if type1 in ONNX_DTYPE_NAMES else ''

        # if parent node is cast node, and same type, delete this one
        pnode = node.inputs[0]
        if pnode.type == 'Cast':
            type2 = pnode.get_attr('to').i
            if type1 == type2:
                for node2 in consumer_nodes:
                    g.replace_input(node2, node2.input[0], node.input[0], 0)
                    q2.append(node2.output[0])
                g.remove_node(node.name)
                return q2

        # otherwise, check consumer cast nodes for a target type
        # that contains more information than current type
        can_reduce = True
        for node2 in consumer_nodes:
            type2 = node2.get_attr('to').i
            type2_name = ONNX_DTYPE_NAMES[type2] if type2 in ONNX_DTYPE_NAMES else ''

            if 'float' in type1_name or type1_name == 'double':
                # high information type. ok to eliminate
                pass
            elif 'int' in type1_name:
                # int* and uint* are mix of high and low information.
                # for safety, keep the current node, unless type2 is bool,
                # in which case it's ok to remove node
                if type1 != type2 and type2_name != 'bool':
                    can_reduce = False
            elif type1_name == 'bool':
                # bool is low information, so don't eliminate
                if type1 != type2:
                    can_reduce = False
            elif type1_name == 'string':
                # can always remove string
                pass
            else:
                # some odd type, keep node
                can_reduce = False
            q2.append(node2.output[0])

        if can_reduce:
            for node2 in consumer_nodes:
                g.replace_input(node2, node2.input[0], node.input[0], 0)
            g.remove_node(node.name)
        return q2

    @staticmethod
    @_register_func("Transpose")
    def _optimize_transpose(g, node, consumer_nodes):
        """remove long chains of transpose ops"""
        t1 = list(node.get_attr('perm').ints)
        q2 = []
        for node2 in consumer_nodes:
            g.replace_input(node2, node2.input[0], node.input[0], 0)
            t2 = list(node2.get_attr('perm').ints)
            new_perm = [t1[i] for i in t2]
            # check if node2 can be removed. otherwise only update
            if new_perm == list(range(len(t2))):
                # both nodes can be deleted
                shape = g.get_shape(node2.output[0])
                dtype = g.get_dtype(node2.output[0])
                node2_consumers = g.find_output_consumers(node2.output[0])
                g.replace_all_inputs(node2.output[0], node.input[0], ops=node2_consumers)
                g.remove_node(node2.name)
                if set(node2.output) & set(g.outputs):
                    g.make_node("Identity", [node.input[0]],
                                outputs=node2.output, shapes=[shape], dtypes=[dtype])
            else:
                node2.set_attr('perm', [t1[i] for i in t2])
                q2.append(node2.output[0])
        g.remove_node(node.name)
        return q2

    @staticmethod
    @_register_func(('Squeeze', 'Unsqueeze'))
    def _optimize_squeeze_unsqueeze(g, node, consumer_nodes):
        """remove pairs of squeeze-unsqueeze nodes"""
        if node.type != 'Squeeze' or len(consumer_nodes) != 1:
            # no need to return any value, since not removing long chain of nodes
            return []

        node2 = consumer_nodes[0]
        if node2.type != 'Unsqueeze':
            return []

        axes_match = False
        if g.opset <= 12 and node.get_attr('axes').ints == node2.get_attr('axes').ints:
            axes_match = True

        # In opset 13, axes is an input. Optional for squeeze op.
        if g.opset >= 13 and len(node.input) == 2:
            if node.input[1] == node2.input[1]:
                axes_match = True
            elif node.inputs[1].is_const() and node2.inputs[1].is_const() and \
                node.inputs[1].get_tensor_value(as_list=True) == node2.inputs[1].get_tensor_value(as_list=True):
                axes_match = True

        # if squeeze followed by unsqueeze is on diff axes, skip
        if not axes_match:
            return []

        # if unsqueeze output is graph output, skip
        if set(node2.output) & set(g.outputs):
            return []

        node2_consumers = g.find_output_consumers(node2.output[0])
        g.replace_all_inputs(node2.output[0], node.input[0], ops=node2_consumers)
        g.remove_node(node.name)
        g.remove_node(node2.name)
        return []

    @staticmethod
    @_register_func(('Conv', 'BatchNormalization'))
    def _optimize_conv_batchnorm_fusion(g, node, consumer_nodes):
        """fuse conv and batchnorm"""
        if node.type != 'Conv' or len(consumer_nodes) != 1:
            # can only fuse 1 conv + batchnorm
            return []

        node2 = consumer_nodes[0]
        if node2.type != 'BatchNormalization':
            return []

        # if batchnorm is a graph output, skip
        if set(node2.output) & set(g.outputs):
            return []

        if not node.inputs[1].is_const():
            return []
        weights = node.inputs[1].get_tensor_value(as_list=False)
        # if not 4D, NCHW skip
        if len(weights.shape) != 4:
            return []

        # optional bias value
        if len(node.inputs) > 2:
            if not node.inputs[2].is_const():
                return []
            bias = node.inputs[2].get_tensor_value(as_list=False)
        else:
            bias = np.array(0, dtype=weights.dtype)

        # scale, offset, mean, var be const, otherwise skip
        if False in [node2.inputs[i].is_const() for i in [1, 2, 3, 4]]:
            return []

        # if bn outputs used elsewhere, cannot fuse
        for i in range(1, len(node2.output)):
            if g.find_output_consumers(node2.output[i]):
                return []

        weights = weights.transpose(2, 3, 1, 0)
        scale = node2.inputs[1].get_tensor_value(as_list=False)
        offset = node2.inputs[2].get_tensor_value(as_list=False)
        mean = node2.inputs[3].get_tensor_value(as_list=False)
        var = node2.inputs[4].get_tensor_value(as_list=False)
        epsilon = node2.get_attr('epsilon').f

        scale_new = scale / np.sqrt(var + epsilon)
        weights_new = weights * scale_new
        weights_new = weights_new.transpose(3, 2, 0, 1)
        bias_new = (bias - mean) * scale_new + offset
        bias_new_const = g.make_const(node.name + '_bias_fused_bn', bias_new.astype(bias.dtype))
        weights_new_const = g.make_const(node.name + '_weights_fused_bn', weights_new.astype(weights.dtype))
        g.replace_inputs(node, [node.input[0], weights_new_const.output[0], bias_new_const.output[0]])

        # fuse conv and bn, delete bn
        node2_output = node2.output[:1]
        node2_shape = g.get_shape(node2.output[0])
        node2_dtype = g.get_dtype(node2.output[0])
        g.remove_node(node2.name)
        # the setter makes a copy
        node.output = node2_output
        g.set_shape(node2_output[0], node2_shape)
        g.set_dtype(node2_output[0], node2_dtype)
        return []

    @staticmethod
    @_register_func('Reshape')
    def _optimize_reshape_reshape(g, node, consumer_nodes):
        """remove sequential reshape nodes"""
        if node.type != 'Reshape' or len(consumer_nodes) != 1:
            return []

        node2 = consumer_nodes[0]
        if node2.type != 'Reshape':
            return []

        g.replace_inputs(node2, [node.input[0], node2.input[1]])
        g.remove_node(node.name)
        return []
