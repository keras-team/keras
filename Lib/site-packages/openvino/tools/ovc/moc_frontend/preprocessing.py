# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log

from openvino.preprocess import PrePostProcessor  # pylint: disable=no-name-in-module,import-error
# pylint: disable=no-name-in-module,import-error
from openvino import Model, Layout, PartialShape
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.moc_frontend.layout_utils import update_layout_to_dict
from openvino.tools.ovc.utils import refer_to_faq_msg


def check_keys_valid(ov_function: Model, dict_to_validate: dict, search_outputs: bool):
    """
    Internal function: checks if keys from cmd line arguments correspond to ov_function's inputs/outputs
    Throws if some key is not found
    Throws if some different keys point to the same actual input/output
    """
    nodes_used = {}
    nodes = ov_function.inputs
    if search_outputs:
        nodes += ov_function.outputs

    # We need to replace all node names from dict to tensor names
    rename_dict = {}
    # Find names for replacing
    for name in dict_to_validate.keys():
        for ov_node in nodes:
            if name in ov_node.get_tensor().get_names():
                break
            elif name == ov_node.get_node().get_friendly_name():
                assert len(ov_node.get_tensor().get_names()) > 0, 'Node must have at least one tensor name'
                new_name = list(ov_node.get_tensor().get_names())[0]
                rename_dict[name] = new_name
                break

    # Replace found node names with tensor names
    for name, new_name in rename_dict.items():
        assert name in dict_to_validate, 'Key {} is not in initial dict'.format(name)
        assert new_name not in dict_to_validate, 'Key {} is already in initial dict'.format(new_name)
        dict_to_validate[new_name] = dict_to_validate[name]
        del dict_to_validate[name]

    # validate the dict
    for name in dict_to_validate.keys():
        node_found = False
        for ov_node in nodes:
            if name in ov_node.get_tensor().get_names():
                if ov_node in nodes_used:
                    raise Error('Key for {} and {} point to same model input/output.'
                                .format(name, nodes_used[ov_node]))
                nodes_used[ov_node] = name
                node_found = True
                break

        if not node_found:
            if not search_outputs:
                raise Error('Input with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))
            else:
                raise Error('Input/Output with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))


def update_layout_is_input_flag(ov_function: Model, layout_values: dict):
    """
    Internal function: updates layout_values with flag whether each layout belongs to input or to output
    """
    for name, layout_value in layout_values.items():
        layout_value['is_input'] = False
        for ov_input in ov_function.inputs:
            if name in ov_input.get_tensor().get_names():
                layout_value['is_input'] = True
                break
    return layout_values


def find_channels_dimension(shape: PartialShape, num_channels: int, name: str, layout_values):
    """
    Internal function. Finds dimension index matching with expected channels number
    Raises error if there is no candidates or number of candidates is > 1
    :param: shape Parameter's partial shape
    :param: num_channels Number of channels to find in shape
    :param: name Parameter's name, used for Error-handling purposes
    :param: layout_values Existing source/target layout items specified by user
    :return: updated layout items with guessed layouts
    """
    if shape.rank.is_dynamic:
        raise Error('Can\'t determine channels dimension for dynamic shape for parameter {}.'
                    .format(name))

    dim_idx_found = -1
    for dim_idx in range(shape.rank.get_length()):
        dim = shape.get_dimension(dim_idx)
        if dim.is_static and dim.get_length() == num_channels:
            if dim_idx_found >= 0:
                raise Error('Can\'t determine channels dimension for {}. '
                            'Input shape is {}, needed channels {}. '
                            'Conflicting dimensions: {} and {}. Please specify layout manually.'
                            .format(name, shape, num_channels, dim_idx_found, dim_idx))
            dim_idx_found = dim_idx
    if dim_idx_found < 0:
        raise Error('Can\'t determine channels dimension for {}. '
                    'Input shape is {}, needed channels {}'
                    .format(name, shape, num_channels))

    # Restrict guessed channels index to particular position depending on tensor shape(3d, 4d, 5d)
    if shape.rank.get_length() == 3:
        # CHW or HWC, possible channels index is 0 or 2
        if dim_idx_found != 0 and dim_idx_found != 2:
            raise Error('Can\'t determine channels dimension for 3D input {} (CHW or HWC) with shape {}. '
                        'Please specify layout containing \'C\' channels manually.'.format(name, shape))
    elif shape.rank.get_length() == 4:
        # NCHW or NHWC, possible channels index is 1 or 3
        if dim_idx_found != 1 and dim_idx_found != 3:
            raise Error('Can\'t determine channels dimension for 4D input {} (NCHW or NHWC) with shape {}. '
                        'Please specify layout containing \'C\' channels manually.'.format(name, shape))
    elif shape.rank.get_length() == 5:
        # NCDHW or NDHWC, possible channels index is 1 or 4
        if dim_idx_found != 1 and dim_idx_found != 4:
            raise Error('Can\'t determine channels dimension for 5D input {} (NCDHW or NDHWC) with shape {}. '
                        'Please specify layout containing \'C\' channels manually.'.format(name, shape))
    else:
        raise Error('Can\'t determine channels dimension for {}D input {} with shape {}.'
                    'Please specify layout containing \'C\' channels manually.'
                    .format(shape.rank.get_length(), name, shape))

    layout_str = "?" * shape.rank.get_length()
    layout_str = layout_str[:dim_idx_found] + 'C' + layout_str[dim_idx_found + 1:]
    layout_values[name] = {
        'source_layout': layout_str,
        'target_layout': None,
        'source_guessed': True,
        'is_input': True
    }
    return layout_values


def update_tensor_names_to_first_in_sorted_list(values_dict: dict, ov_function: Model):
    if not isinstance(values_dict, dict):
        return values_dict
    updated_dict = {}
    used_nodes = {}
    for name, value in values_dict.items():
        input_found = False
        for input in ov_function.inputs:
            tensor_names = list(input.names)
            tensor_names.sort()
            if not (name in tensor_names or name == input.node.get_friendly_name()):
                continue
            if input in used_nodes:
                raise Error("Tensor names {} and {} refer to the same node.".format(name, used_nodes[input]))
            used_nodes.update({input: name})
            updated_dict[tensor_names[0]] = value
            input_found = True
            break
        if not input_found:
            raise Error('Input with name {} wasn\'t found! {}'.format(name, refer_to_faq_msg(83)))

    return updated_dict


def apply_preprocessing(ov_function: Model, argv: argparse.Namespace):
    """
    Applies pre-processing of model inputs by adding appropriate operations
    On return, 'ov_function' object will be updated
    Expected 'argv.mean_scale_values' formats examples:
        a) Dict: {'inputName': {'mean': [1., 2., 3.], 'scale': [2., 4., 8.]}}
        b) List: list(np.array([(np.array([1., 2., 3.]), np.array([2., 4., 6.])),
                     (np.array([7., 8., 9.]), np.array([5., 6., 7.])))
    Expected 'argv.layout_values' format examples:
        a) Specific layouts for inputs and outputs
        { 'input1': {
                 'source_layout': 'nchw',
                 'target_layout': 'nhwc'
             },
             'output2': {
                 'source_layout': 'nhwc'
             }
        }
        b) Layout for single input: {'': {'source_layout': 'nchw'}}
    :param: ov_function OV function for applying mean/scale pre-processing
    :param: argv Parsed command line arguments
    """
    prep = PrePostProcessor(ov_function)

    layout_values = {}
    if 'layout_values' in argv and argv.layout_values:
        layout_values = update_layout_to_dict(ov_function.inputs, argv.layout_values,
                                              lambda ov_input: ov_input.get_tensor().get_names())

    check_keys_valid(ov_function=ov_function, dict_to_validate=layout_values, search_outputs=True)

    layout_values = update_layout_is_input_flag(ov_function, layout_values)

    for node_name, layout_value in layout_values.items():
        if layout_value.get('source_layout'):
            if layout_value.get('is_input'):
                prep.input(node_name).model().set_layout(Layout(layout_value['source_layout']))
            else:
                prep.output(node_name).model().set_layout(Layout(layout_value['source_layout']))
        if layout_value.get('target_layout'):
            if layout_value.get('is_input'):
                prep.input(node_name).tensor().set_layout(Layout(layout_value['target_layout']))
            else:
                prep.output(node_name).tensor().set_layout(Layout(layout_value['target_layout']))

    # Apply pre-processing builder to a function
    ov_function = prep.build()

    # Remove guessed layout values from ov_function (these values shall not be serialized to IR
    for node_name, layout_value in layout_values.items():
        if layout_value.get('source_guessed') and \
                not layout_value.get('target_layout'):
            # search for parameter object
            for idx, ov_input in enumerate(ov_function.inputs):
                if node_name in ov_input.get_tensor().get_names():
                    log.debug('Clearing guessed layout {} for {}'
                              .format(layout_value['source_layout'], node_name))
                ov_function.get_parameters()[idx].layout = Layout()
