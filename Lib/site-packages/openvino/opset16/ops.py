# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset16."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes

_get_node_factory_opset16 = partial(_get_node_factory, "opset16")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def identity(
    data: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Identity operation is used as a placeholder. It creates a copy of the input to forward to the output.

    :param data: Tensor with data.

    :return: The new node performing Identity operation.
    """
    return _get_node_factory_opset16().create(
        "Identity",
        as_nodes(data, name=name),
        {},
    )
