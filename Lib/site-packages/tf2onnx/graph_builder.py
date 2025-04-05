# SPDX-License-Identifier: Apache-2.0


"""
tf2onnx.graph_helper - class to help building graph, such as helping to make complex node
"""

import numpy as np
from tf2onnx import utils, logging


# pylint: disable=missing-docstring


logger = logging.getLogger(__name__)


class GraphBuilder(object):
    """help to build graph"""
    def __init__(self, graph):
        self._g = graph

    @property
    def graph(self):
        return self._g

    def make_slice(self, kwargs, name=None, shapes=None, dtypes=None, return_node=False):
        """
        slice changes its schema at opset 10: it treats some attributes as dynamic input
        so this function has to process inputs according to graph's opset version
        to get "inputs" and "attr" to feed "make_node"
        kwargs: key could be ["data", "starts", "ends", "axes", "steps", "outputs"].
        """
        outputs = kwargs.pop("outputs", None)

        if self.graph.opset < 10:
            # "data" is string
            # "starts", "ends" and "axes" are attributes, and "axes" is optional.
            data = kwargs.pop("data")
            starts = self.convert_to_attribute(kwargs.pop("starts"))
            ends = self.convert_to_attribute(kwargs.pop("ends"))
            axes = self.convert_to_attribute(kwargs.pop("axes", None), is_optional=True)
            attr = {"starts": starts, "ends": ends, "axes": axes}
            inputs = [data]
        else:
            # slice-10 has 3 required inputs "data", "starts", "ends"l
            # and 2 optional inputs "axes", "steps"
            # input sequence should be "data", "starts", "ends", "axes", "steps"
            attr = {}
            data = kwargs.pop("data")
            starts = self.convert_to_input(kwargs.pop("starts"), "const_starts", dtype=np.int64)
            ends = self.convert_to_input(kwargs.pop("ends"), "const_ends", dtype=np.int64)
            axes = self.convert_to_input(kwargs.pop("axes", None), "const_axes", is_optional=True, dtype=np.int64)
            steps = self.convert_to_input(kwargs.pop("steps", None), "const_steps", is_optional=True, dtype=np.int64)
            inputs = [data, starts, ends, axes, steps]

        # pro-process inputs and attr
        utils.make_sure(not kwargs, "kwargs contains un-used key")

        new_attr = {}
        for key, val in attr.items():
            if val is not None:
                new_attr[key] = val
        attr = new_attr

        for ind, val in enumerate(inputs):
            if val is None:
                inputs[ind] = utils.ONNX_EMPTY_INPUT  # empty string means no connection in ONNX
        # remove tailing ""
        while inputs[-1] == utils.ONNX_EMPTY_INPUT:
            inputs = inputs[:-1]

        if self.graph.opset >= 10:
            dtype = self.graph.get_dtype(inputs[1])
            for input_data in inputs[1:]:
                if input_data != utils.ONNX_EMPTY_INPUT:
                    utils.make_sure(dtype == self.graph.get_dtype(input_data), "dtype should be same")

        node = self.graph.make_node(op_type="Slice", inputs=inputs, attr=attr, name=name,
                                    outputs=outputs, shapes=shapes, dtypes=dtypes)
        if return_node:
            return node
        return node.output[0]

    def _make_reduce_op(self, op_name, since_opset, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceSum changes its schema at opset 13: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        After opset 18, all of Reduce* ops has taken the same change.
        """
        outputs = kwargs.pop("outputs", None)

        if self.graph.opset < since_opset:
            data = kwargs.pop("data")
            axes = self.convert_to_attribute(kwargs.pop("axes", None), is_optional=True)
            keepdims = kwargs.pop("keepdims", 1)
            noop_with_empty_axes = kwargs.pop("noop_with_empty_axes", 0)
            if noop_with_empty_axes == 0 and axes == []:
                axes = None
            attr = {"axes": axes, "keepdims": keepdims}
            inputs = [data]
        else:
            keepdims = kwargs.pop("keepdims", 1)
            noop_with_empty_axes = kwargs.pop("noop_with_empty_axes", 0)
            data = self.convert_to_input(kwargs.pop("data"), "const_data")
            attr = {"keepdims": keepdims, "noop_with_empty_axes": noop_with_empty_axes}
            inputs = [data]
            if "axes" in kwargs.keys():
                axes = self.convert_to_input(kwargs.pop("axes", None), "const_axes", is_optional=True, dtype=np.int64)
                inputs.append(axes)

        utils.make_sure(not kwargs, "kwargs contains un-used key")

        new_attr = {}
        for key, val in attr.items():
            if val is not None:
                new_attr[key] = val
        attr = new_attr

        return self.graph.make_node(op_type=op_name, inputs=inputs, attr=attr, name=name,
                                    outputs=outputs, shapes=shapes, dtypes=dtypes,
                                    op_name_scope=op_name_scope).output[0]

    def make_reduce_max(self, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceMax changes its schema at opset 18: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        """

        return self._make_reduce_op("ReduceMax", 18, kwargs, name, shapes, dtypes, op_name_scope)

    def make_reduce_mean(self, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceMean changes its schema at opset 18: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        """

        return self._make_reduce_op("ReduceMean", 18, kwargs, name, shapes, dtypes, op_name_scope)

    def make_reduce_min(self, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceMin changes its schema at opset 18: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        """

        return self._make_reduce_op("ReduceMin", 18, kwargs, name, shapes, dtypes, op_name_scope)

    def make_reduce_prod(self, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceProd changes its schema at opset 18: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        """

        return self._make_reduce_op("ReduceProd", 18, kwargs, name, shapes, dtypes, op_name_scope)

    def make_reduce_sum(self, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceSum changes its schema at opset 13: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        """

        return self._make_reduce_op("ReduceSum", 13, kwargs, name, shapes, dtypes, op_name_scope)

    def make_reduce_sum_square(self, kwargs, name=None, shapes=None, dtypes=None, op_name_scope=None):
        """
        ReduceSumSquare changes its schema at opset 13: it treats some axes as dynamic input
        kwargs: key could be ["data", "axes", "keepdims", "noop_with_empty_axes", "outputs"].
        """

        return self._make_reduce_op("ReduceSumSquare", 18, kwargs, name, shapes, dtypes, op_name_scope)

    def make_squeeze(self, kwargs, name=None, shapes=None, dtypes=None, return_node=False, op_name_scope=None):
        """
        Squeeze changes its schema at opset 13: it treats axes as a dynamic input
        kwargs: key could be ["data", "axes"].
        """
        outputs = kwargs.pop("outputs", None)

        if self.graph.opset < 13:
            data = kwargs.pop("data")
            axes = self.convert_to_attribute(kwargs.pop("axes", None), is_optional=True)
            attr = {"axes": axes}
            inputs = [data]
        else:
            data = kwargs.pop("data")
            axes = self.convert_to_input(kwargs.pop("axes", None), "const_axes", is_optional=True, dtype=np.int64)
            attr = {}
            inputs = [data, axes]

        utils.make_sure(not kwargs, "kwargs contains un-used key")

        new_attr = {}
        for key, val in attr.items():
            if val is not None:
                new_attr[key] = val
        attr = new_attr

        for ind, val in enumerate(inputs):
            if val is None:
                inputs[ind] = utils.ONNX_EMPTY_INPUT  # empty string means no connection in ONNX
        # remove tailing ""
        while inputs[-1] == utils.ONNX_EMPTY_INPUT:
            inputs = inputs[:-1]

        node = self.graph.make_node(op_type="Squeeze", inputs=inputs, attr=attr, name=name,
                                    outputs=outputs, shapes=shapes, dtypes=dtypes,
                                    op_name_scope=op_name_scope)
        if return_node:
            return node
        return node.output[0]

    def make_unsqueeze(self, kwargs, name=None, shapes=None, dtypes=None, return_node=False, op_name_scope=None):
        """
        Unsqueeze changes its schema at opset 13: it treats axes as a dynamic input
        kwargs: key could be ["data", "axes"].
        """
        outputs = kwargs.pop("outputs", None)

        if self.graph.opset < 13:
            data = kwargs.pop("data")
            axes = self.convert_to_attribute(kwargs.pop("axes", None), is_optional=True)
            attr = {"axes": axes}
            inputs = [data]
        else:
            data = kwargs.pop("data")
            axes = self.convert_to_input(kwargs.pop("axes", None), "const_axes", is_optional=True, dtype=np.int64)
            attr = {}
            inputs = [data, axes]

        utils.make_sure(not kwargs, "kwargs contains un-used key")

        new_attr = {}
        for key, val in attr.items():
            if val is not None:
                new_attr[key] = val
        attr = new_attr

        for ind, val in enumerate(inputs):
            if val is None:
                inputs[ind] = utils.ONNX_EMPTY_INPUT  # empty string means no connection in ONNX
        # remove tailing ""
        while inputs[-1] == utils.ONNX_EMPTY_INPUT:
            inputs = inputs[:-1]

        node = self.graph.make_node(op_type="Unsqueeze", inputs=inputs, attr=attr, name=name,
                                    outputs=outputs, shapes=shapes, dtypes=dtypes,
                                    op_name_scope=op_name_scope)
        if return_node:
            return node
        return node.output[0]

    def convert_to_input(self, tensor, const_name, is_optional=False, dtype=None):
        """in ONNX, input shold come from node, so it must be a string"""
        if is_optional and tensor is None:
            return None

        utils.make_sure(tensor is not None, "input is required so it couldn't be None")

        res = tensor
        if isinstance(tensor, list):
            res = self.graph.make_const(utils.make_name(const_name), np.array(tensor, dtype)).output[0]

        utils.make_sure(isinstance(res, str), "input is a dynamic input, so a str is needed")

        return res

    def convert_to_attribute(self, tensor, is_optional=False):
        if is_optional and tensor is None:
            return None

        utils.make_sure(tensor is not None, "input is required so it couldn't be None")

        res = tensor
        if isinstance(tensor, str):
            const_node = self.graph.get_node_by_output(tensor)
            res = const_node.get_tensor_value(as_list=True)

        utils.make_sure(isinstance(res, list), "input is an attr, so a list is needed")

        return res
