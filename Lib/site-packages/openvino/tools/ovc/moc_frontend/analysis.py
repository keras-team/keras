# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

from openvino import PartialShape, Model, Type  # pylint: disable=no-name-in-module,import-error
from openvino.utils.types import get_dtype  # pylint: disable=no-name-in-module,import-error


def json_model_analysis_dump(framework_model: Model):
    def dump_partial_shape(shape: PartialShape):
        if shape.rank.is_dynamic:
            return 'None'
        return [dim.get_length() if dim.is_static else 0 for dim in shape]

    def dump_element_type(ov_type: Type):
        try:
            return str(get_dtype(ov_type))
        except:
            return 'None'

    json_dump = {}
    json_dump['inputs'] = {}
    for param in framework_model.get_parameters():
        param_name = param.get_friendly_name()
        json_dump['inputs'][param_name] = {}
        json_dump['inputs'][param_name]['shape'] = dump_partial_shape(param.get_partial_shape())
        json_dump['inputs'][param_name]['data_type'] = dump_element_type(param.get_element_type())
        json_dump['inputs'][param_name]['value'] = 'None'  # not supported in 22.1

    json_dump['intermediate'] = {}
    # TODO: extend model analysis dump for operations with body graphs (If, Loop, and TensorIterator)
    for op in filter(lambda node: node.type_info.name != "NullNode", framework_model.get_ordered_ops()):
        for out_idx in range(op.get_output_size()):
            output = op.output(out_idx)
            tensor_name = output.get_any_name()
            json_dump['intermediate'][tensor_name] = {}
            json_dump['intermediate'][tensor_name]['shape'] = dump_partial_shape(output.get_partial_shape())
            json_dump['intermediate'][tensor_name]['data_type'] = dump_element_type(output.get_element_type())
            json_dump['intermediate'][tensor_name]['value'] = 'None'  # not supported in 22.1

    json_model_analysis_print(json_dump)


def json_model_analysis_print(json_dump: str):
    print(json.dumps(json_dump))
