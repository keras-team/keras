# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from openvino import PartialShape  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.utils import refer_to_faq_msg


def update_layout_to_dict(inputs: list, layout: [list, dict], get_names_func: Callable):
    """
    The function prepares layout values in the dictionary with items of the format:
    { node_name : {'source_layout': 'NHWC', 'target_layout': 'NCHW'} }
    """
    if isinstance(layout, dict):
        if '' in layout:
            input_names = [list(get_names_func(cur_input))[0] for cur_input in inputs]
            if len(input_names) > 1:
                raise Error('Layout without name can be specified for models with only one input, '
                            'but provided model has {} inputs: \'{}\'. '
                            'Please specify explicitly input/output name for "layout" option'
                            .format(len(input_names), input_names))
            layout = {
                input_names[0]: {
                    'source_layout': layout[''].get('source_layout'),
                    'target_layout': layout[''].get('target_layout')
                }
            }
        return layout
    if isinstance(layout, list):
        if len(layout) != len(inputs):
            raise Error('Numbers of inputs and layout values do not match. ' + refer_to_faq_msg(61))
        layout_dict = {}
        for idx, cur_input in enumerate(inputs):
            names_list = list(get_names_func(cur_input))
            assert len(names_list) > 0, "No names for input"
            node_name = names_list[0]
            layout_dict.update(
                {
                    node_name: layout[idx]
                }
            )
        return layout_dict
    raise Error("Unknown layout type. Expected dict, list. Got {}".format(type(layout)))


def get_dimension_index_by_label(input_shape: PartialShape, input_names: list, layout_dict: [dict],
                                 dimension_label: str, default_dim: int):
    """
    The function returns index of the dimension pointed in the layout
    and a flag indicating if the index is chosen by default.
    For example, the index for 'D' dimension in "NHWDC" layout is 3.
    """
    if input_shape.rank.is_static and input_shape.rank.get_length() == 0:
        # in case a scalar, batch dimension is not defined
        return None, False

    # search for the corresponding layout
    for name, layout_value in layout_dict.items():
        if name in input_names:
            layout = layout_value.get('source_layout', None)
            if layout is None:
                return default_dim, True
            from openvino import Layout  # pylint: disable=no-name-in-module,import-error
            layout_parsed = Layout(layout)
            if layout_parsed.has_name(dimension_label):
                return layout_parsed.get_index_by_name(dimension_label), False
            else:
                # if the layout is specified and the required dimension label is not found, the batch is unknown
                return None, False

    return default_dim, True
