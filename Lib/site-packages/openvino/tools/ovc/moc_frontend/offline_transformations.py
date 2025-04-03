# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

from openvino.tools.ovc.error import Error


def get_new_placeholder_name(node_id: str, is_out_port: bool = False, port: int = 0):
    """
    Forms a name of new placeholder created by cutting a graph
    :param node_id: a node name that is cut
    :param is_out_port: it is True iff output port is cut
    :param port: a port number
    :return: a name of new placeholder created by cutting a graph
    """
    port_type = '_out' if is_out_port else ''
    return '{}/placeholder{}_port_{}'.format(node_id, port_type, port)


def create_params_with_custom_types(packed_user_shapes: [None, dict]):
    """
    Compute a list of placeholder names for which an user specifies custom type
    :param packed_user_shapes: packed data that contains input node names,
    their port numbers, shapes and data types
    :return: a list of placeholder names for which an user specifies custom type
    Example of packed_user_shapes dictionary:
    packed_user_shapes =
    {
        'node_ID':
            [
                {'shape': None, 'in': 0},
                {'shape': None, 'in': 1},
            ],
        'node_1_ID':
            [
                {'shape': [1, 227, 227, 3], 'port': None, 'data_type': np.int32}
            ],
        'node_2_ID':
            [
                {'shape': None, 'out': 3}
            ]
    }
    For which the function returns a list ['node_1_ID'] because this node only has custom data type
    """
    if packed_user_shapes is None:
        return []

    params_with_custom_types = []
    for input_name in packed_user_shapes:
        for desc in packed_user_shapes[input_name]:
            p_name = input_name
            if 'port' in desc and desc['port'] is None:  # neither input nor output port specified
                user_defined_type = desc.get('data_type', None)
            else:  # need to check the particular port the Parameter was created for
                p_name = get_new_placeholder_name(input_name, 'out' in desc,
                                                  desc['out'] if 'out' in desc else desc['in'])
                user_defined_type = desc.get('data_type', None)
            if user_defined_type is not None:
                params_with_custom_types.append(p_name)
    return params_with_custom_types


def get_available_transformations():
    try:
        from openvino._offline_transformations import \
            apply_low_latency_transformation  # pylint: disable=import-error,no-name-in-module
        from openvino._offline_transformations import \
            apply_make_stateful_transformation  # pylint: disable=import-error,no-name-in-module
        from openvino._offline_transformations import \
            apply_pruning_transformation  # pylint: disable=import-error,no-name-in-module
        return {
            'MakeStateful': apply_make_stateful_transformation,
            'LowLatency2': apply_low_latency_transformation,
            'Pruning': apply_pruning_transformation,
        }
    except Exception as e:
        return {}


# net should be openvino.Model type, but OV Engine is still optional dependency
def apply_user_transformations(func: object, transforms: list):
    available_transformations = get_available_transformations()

    for name, args in transforms:
        if name not in available_transformations.keys():
            raise Error("Transformation {} is not available.".format(name))

        available_transformations[name](func, **args)


def apply_moc_legacy_transformations(func: object, params_with_custom_types: List[str]):
    from openvino._offline_transformations import \
        apply_moc_legacy_transformations  # pylint: disable=import-error,no-name-in-module
    apply_moc_legacy_transformations(func, params_with_custom_types)


def compress_model(func: object):
    from openvino._offline_transformations import \
        compress_model_transformation  # pylint: disable=import-error,no-name-in-module
    compress_model_transformation(func)


def apply_fused_names_cleanup(func: object):
    from openvino._offline_transformations import \
        apply_fused_names_cleanup  # pylint: disable=import-error,no-name-in-module
    apply_fused_names_cleanup(func)
