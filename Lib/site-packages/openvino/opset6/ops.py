# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

"""Factory functions for all openvino ops."""
from typing import Optional, Union

from functools import partial, singledispatch

from openvino import Node, Type, PartialShape, Output, Shape
from openvino.op import assign, Constant, Parameter
from openvino.op import read_value as _read_value
from openvino.op.util import VariableInfo, Variable
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.decorators import nameable_op, overloading
from openvino.utils.types import (
    NodeInput,
    NumericType,
    TensorShape,
    as_node,
    as_nodes,
    get_element_type,
)

_get_node_factory_opset6 = partial(_get_node_factory, "opset6")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def ctc_greedy_decoder_seq_len(
    data: NodeInput,
    sequence_length: NodeInput,
    blank_index: Optional[NodeInput] = None,
    merge_repeated: bool = True,
    classes_index_type: str = "i32",
    sequence_length_type: str = "i32",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs CTCGreedyDecoderSeqLen.

    :param data:            The input 3D tensor. Shape: [batch_size, seq_length, num_classes]
    :param sequence_length: Input 1D tensor with sequence length. Shape: [batch_size]
    :param blank_index:     Scalar or 1D tensor with specifies the class index to use for the blank class.
                            Optional parameter. Default value is num_classes-1.
    :return:                The new node which performs CTCGreedyDecoderSeqLen.
    """
    if blank_index is not None:
        inputs = as_nodes(data, sequence_length, blank_index, name=name)
    else:
        inputs = as_nodes(data, sequence_length, name=name)

    attributes = {
        "merge_repeated": merge_repeated,
        "classes_index_type": classes_index_type,
        "sequence_length_type": sequence_length_type,
    }

    return _get_node_factory_opset6().create("CTCGreedyDecoderSeqLen", inputs, attributes)


@nameable_op
def gather_elements(
    data: NodeInput,
    indices: NodeInput,
    axis: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GatherElements.

    :param data:       N-D tensor with data for gathering
    :param indices:    N-D tensor with indices by which data is gathered
    :param axis:       axis along which elements are gathered
    :return:           The new node which performs GatherElements
    """
    inputs = as_nodes(data, indices, name=name)

    attributes = {
        "axis": axis,
    }

    return _get_node_factory_opset6().create("GatherElements", inputs, attributes)


@nameable_op
def mvn(
    data: Node,
    axes: Node,
    normalize_variance: bool,
    eps: float,
    eps_mode: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs MeanVarianceNormalization (MVN).

    :param data: The node with data tensor.
    :param axes: The node with axes to reduce on.
    :param normalize_variance: Denotes whether to perform variance normalization.
    :param eps: The number added to the variance to avoid division by zero
               when normalizing the value. Scalar value.
    :param eps_mode: how eps is applied (`inside_sqrt` or `outside_sqrt`)
    :param name: Optional output node name.
    :return: The new node performing a MVN operation on input tensor.
    """
    inputs = as_nodes(data, axes, name=name)

    attributes = {
        "normalize_variance": normalize_variance,
        "eps": eps,
        "eps_mode": eps_mode,
    }

    return _get_node_factory_opset6().create("MVN", inputs, attributes)


@overloading(Union[Node, Output, int, float, np.ndarray], str, Optional[Union[type, np.dtype, Type, str]],
             Optional[Union[TensorShape, Shape, PartialShape]], Optional[str])
@nameable_op
def read_value(init_value: Union[Node, Output, int, float, np.ndarray],
               variable_id: str,
               variable_type: Optional[Union[type, np.dtype, Type, str]] = None,
               variable_shape: Optional[Union[TensorShape, Shape, PartialShape]] = None,
               name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param init_value:   Node producing a value to be returned instead of an unassigned variable.
    :param variable_id:  Id of a variable to be read.
    :param variable_type:   Optional type to be set into Variable.
    :param variable_shape:  Optional shape to be set into Variable.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    info = VariableInfo()
    info.variable_id = variable_id

    if variable_type is not None:
        if not isinstance(variable_type, Type) and not isinstance(variable_type, str):
            info.data_type = get_element_type(variable_type)
        else:
            info.data_type = variable_type
    else:
        info.data_type = Type.dynamic

    if variable_shape is not None:
        info.data_shape = PartialShape(variable_shape)
    else:
        info.data_shape = PartialShape.dynamic()

    var_from_info = Variable(info)
    return _read_value(new_value=as_node(init_value, name=name), variable=var_from_info)


@overloading(str, Optional[Union[type, np.dtype, Type, str]], Optional[Union[TensorShape, Shape, PartialShape]], Optional[str])  # type: ignore
@nameable_op
def read_value(variable_id: str,  # noqa: F811
               variable_type: Optional[Union[type, np.dtype, Type, str]] = None,
               variable_shape: Optional[Union[TensorShape, Shape, PartialShape]] = None,
               name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param variable_id:  Id of a variable to be read.
    :param variable_type:   Optional type to be set into Variable.
    :param variable_shape:  Optional shape to be set into Variable.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    info = VariableInfo()
    info.variable_id = variable_id

    if variable_type is not None:
        if not isinstance(variable_type, Type) and not isinstance(variable_type, str):
            info.data_type = get_element_type(variable_type)
        else:
            info.data_type = variable_type
    else:
        info.data_type = Type.dynamic

    if variable_shape is not None:
        info.data_shape = PartialShape(variable_shape)
    else:
        info.data_shape = PartialShape.dynamic()

    var_from_info = Variable(info)

    return _read_value(var_from_info)


@overloading(Variable, Optional[str])    # type: ignore
@nameable_op
def read_value(ov_variable: Variable,  # noqa: F811
               name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param ov_variable:  Variable to be read.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    return _read_value(ov_variable)


@overloading(Union[Node, Output, int, float, np.ndarray], Variable, Optional[str])  # type: ignore
@nameable_op
def read_value(init_value: Union[Node, Output, int, float, np.ndarray],  # noqa: F811
               ov_variable: Variable,
               name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param init_value:   Optional node producing a value to be returned instead of an unassigned variable.
    :param ov_variable:  Variable to be read.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    return _read_value(as_node(init_value, name=name), ov_variable)
