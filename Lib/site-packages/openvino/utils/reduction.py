# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Iterable, Optional

from openvino import Node


def get_reduction_axes(node: Node, reduction_axes: Optional[Iterable[int]]) -> Iterable[int]:
    """Get reduction axes if it is None and convert it to set if its type is different.

    If reduction_axes is None we default to reduce all axes.

    :param node: The node we fill reduction axes for.
    :param reduction_axes: The collection of indices of axes to reduce. May be None.

    returns: Set filled with indices of axes we want to reduce.
    """
    if reduction_axes is None:
        reduction_axes = set(range(len(node.shape)))

    if type(reduction_axes) is not set:
        reduction_axes = set(reduction_axes)
    return reduction_axes
