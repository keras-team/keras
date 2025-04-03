# SPDX-License-Identifier: Apache-2.0


"""symbolic executor
   Computes a part of the graph symbolically using SymbolicTensorElements
"""

import numpy as np
from tf2onnx import utils


# pylint: disable=logging-not-lazy,unused-argument,missing-docstring


class SymbolicTensorElement:
    """
    Helps with symbolic execution of the graph, in particular tensors representing shapes. Supports multiplication
    and tensor ops.
    """
    def __init__(self, terms, constant):
        # Terms is a list representing variables
        self.terms = terms
        self.constant = constant
        if self.constant == 0:
            self.terms = []

    def __mul__(self, other):
        if isinstance(other, SymbolicTensorElement):
            # Concat terms, multiply constant
            return SymbolicTensorElement(self.terms + other.terms, self.constant * other.constant)
        # Other term is a constant
        return SymbolicTensorElement(self.terms, self.constant * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def is_const(self):
        return len(self.terms) == 0

    def is_one(self):
        return len(self.terms) == 0 and self.constant == 1

    def is_single_var(self):
        return len(self.terms) == 1 and self.constant == 1

    def has_multiple_terms(self):
        return not self.is_const() and not self.is_single_var()

    def get_reshape_dim(self, i, offset):
        if self.is_const():
            return self.constant
        if self.get_offset(i) == offset:
            return 0
        return -1

    @staticmethod
    def from_const(constant):
        return SymbolicTensorElement([], constant)

    @staticmethod
    def from_variable(variable):
        return SymbolicTensorElement([variable], 1)

    @staticmethod
    def from_value(value):
        if isinstance(value, SymbolicTensorElement):
            return value
        return SymbolicTensorElement.from_const(value)

    @staticmethod
    def np_array(np_array):
        return np.vectorize(SymbolicTensorElement.from_value)(np_array)

class SymbolicExecutionException(Exception):
    pass

class SymbolicExecutor:
    def __init__(self, graph):
        self.graph = graph
        self.op_map = {
            "Unsqueeze": self.compute_squeeze_unsqueeze,
            "Squeeze": self.compute_squeeze_unsqueeze,
            "Gather": self.compute_gather,
            "Mul": self.compute_mul,
            "ReduceProd": self.compute_reduceprod,
            "Slice": self.compute_slice,
            "Cast": self.compute_cast,
            "Concat": self.compute_concat,
            "Const": self.compute_const
        }

    def compute_outputs(self, outputs, feed_dict):
        """Given a map of inputs to np arrays, outputs a list of np arrays of SymbolicTensorElements"""
        nodes_to_compute = self.plan_computation(outputs, feed_dict)
        if nodes_to_compute is None:
            return None
        results = feed_dict.copy()
        for node in nodes_to_compute:
            try:
                results.update(self.compute_node(node, results))
            except Exception as e:
                raise SymbolicExecutionException(str(e))
        # Intermediate results might be non-symbolic numpy arrays
        return [SymbolicTensorElement.np_array(results[out]) for out in outputs]

    def plan_computation(self, outputs, feed_dict):
        nodes = list(set(self.graph.get_node_by_output(out) for out in outputs))
        sorted_nodes = []
        while nodes:
            n = nodes.pop()
            if n.type not in self.op_map:
                raise SymbolicExecutionException("Unsupported op %s" % n.type)
            sorted_nodes.append(n)
            for inp, inp_name in zip(n.inputs, n.input):
                if inp_name != '' and inp_name not in feed_dict:
                    nodes.append(inp)
        return sorted_nodes[::-1]

    def compute_node(self, node, feed_dict):
        results = self.op_map[node.type](node, feed_dict)
        return {out: np.array(res) for out, res in zip(node.output, results)}

    def compute_const(self, node, feed_dict):
        return [node.get_tensor_value(as_list=False)]

    def compute_squeeze_unsqueeze(self, node, feed_dict):
        inp1 = feed_dict[node.input[0]]
        if self.graph.opset < 13:
            axes = node.get_attr_value("axes")
        else:
            axes = feed_dict[node.input[1]].tolist()
        shape = inp1.shape
        handler = self.compute_unsqueeze_shape if node.type == "Unsqueeze" else self.compute_squeeze_shape
        new_shape = handler(shape, axes)
        return [inp1.reshape(new_shape)]

    def compute_cast(self, node, feed_dict):
        inp = feed_dict[node.input[0]]
        if inp.dtype == object:
            return [inp]
        np_dtype = utils.ONNX_TO_NUMPY_DTYPE[node.get_attr("to").i]
        return [inp.astype(np_dtype)]

    def compute_mul(self, node, feed_dict):
        return [feed_dict[node.input[0]] * feed_dict[node.input[1]]]

    def compute_reduceprod(self, node, feed_dict):
        inp = feed_dict[node.input[0]]
        if self.graph.opset < 18:
            axes = node.get_attr_value("axes")
        else:
            axes = feed_dict[node.input[1]]
        keepdims = node.get_attr_value("keepdims", 1)
        return [np.prod(inp, axis=tuple(axes), keepdims=keepdims)]

    def compute_slice(self, node, feed_dict):
        inps = [feed_dict[inp] if inp != '' else None for inp in node.input]
        if self.graph.opset >= 10:
            while len(inps) < 5:
                inps.append(None)
            data, starts, ends, axes, steps = inps
        else:
            data = inps[0]
            starts = node.get_attr_value("starts")
            ends = node.get_attr_value("ends")
            axes = node.get_attr_value("axes")
            steps = None
        rank = len(data.shape)
        ndims = len(starts)
        if axes is None:
            axes = list(range(ndims))
        if steps is None:
            steps = [1] * ndims
        slices = [slice(None, None, None) for _ in range(rank)]
        for axis, start, end, step in zip(axes, starts, ends, steps):
            slices[axis] = slice(start, end, step)
        return [data[tuple(slices)]]

    def compute_concat(self, node, feed_dict):
        axis = node.get_attr_value("axis")
        inps = [feed_dict[inp] for inp in node.input]
        return [np.concatenate(inps, axis=axis)]

    def compute_gather(self, node, feed_dict):
        data = feed_dict[node.input[0]]
        indices = feed_dict[node.input[1]]
        if indices.dtype == object:
            raise SymbolicExecutionException("Gather requires non-symbolic indices")
        axis = node.get_attr_value("axis", 0)
        return [np.take(data, indices, axis=axis)]

    def compute_unsqueeze_shape(self, shape_in, axes):
        dims_out = len(shape_in) + len(axes)
        axes = [i if i >= 0 else i + dims_out for i in axes]
        shape_in = iter(shape_in)
        shape_out = [None] * dims_out
        for ind in axes:
            shape_out[ind] = 1
        for ind, val in enumerate(shape_out):
            if val is None:
                shape_out[ind] = next(shape_in)
        return shape_out

    def compute_squeeze_shape(self, shape_in, axes):
        axes = [i if i >= 0 else i + len(axes) for i in axes]
        shape_out = []
        for ind, val in enumerate(shape_in):
            if ind not in axes:
                shape_out.append(val)
        return shape_out
