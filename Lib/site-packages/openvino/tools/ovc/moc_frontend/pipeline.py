# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import sys
from typing import List

import numpy as np
import os

from openvino.frontend import FrontEnd, InputModel, NotImplementedFailure, \
    Place  # pylint: disable=no-name-in-module,import-error
from openvino import PartialShape, Type  # pylint: disable=no-name-in-module,import-error
from openvino.utils.types import get_element_type, \
    get_numpy_ctype  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.moc_frontend.analysis import json_model_analysis_dump
from openvino.tools.ovc.moc_frontend.extractor import fe_user_data_repack, convert_params_lists_to_dicts, \
    fe_output_user_data_repack
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.utils import np_map_cast, mo_array


def get_enabled_and_disabled_transforms():
    """
    :return: tuple of lists with force enabled and disabled id of transformations.
    """
    disabled_transforms = os.environ['MO_DISABLED_TRANSFORMS'] if 'MO_DISABLED_TRANSFORMS' in os.environ else ''
    enabled_transforms = os.environ['MO_ENABLED_TRANSFORMS'] if 'MO_ENABLED_TRANSFORMS' in os.environ else ''

    assert isinstance(enabled_transforms, str)
    assert isinstance(disabled_transforms, str)

    disabled_transforms = disabled_transforms.split(',')
    enabled_transforms = enabled_transforms.split(',')

    return enabled_transforms, disabled_transforms


def raise_exception_for_input_output_cut(model_inputs_or_outputs: List[Place], new_nodes: List[dict], is_input: bool):
    for new_node in new_nodes:
        node = new_node['node']

        if not any([item.is_equal(node) for item in model_inputs_or_outputs]):
            if is_input:
                raise Exception("Name {} is not found among model inputs.".format(new_node['input_name']))
            else:
                raise Exception("Name {} is not found among model outputs.".format(new_node['output_name']))


def moc_pipeline(argv: argparse.Namespace, moc_front_end: FrontEnd):
    """
    Load input model and convert it to nGraph function
    :param: argv: parsed command line arguments
    :param: moc_front_end: Loaded Frontend for converting input model
    :return: converted nGraph function ready for serialization
    """

    share_weights = getattr(argv, 'share_weights', True)  # FIXME: Should be controlled by default value
    if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 2:
        # frozen format with v1 checkpoints
        input_model = moc_front_end.load([part for part in argv.input_model], share_weights)
    else:
        input_model = moc_front_end.load(argv.input_model, share_weights)

    '''elif argv.input_meta_graph: # TODO: Cover this case
        input_model = moc_front_end.load(argv.input_meta_graph, share_weights)
        if argv.output:
            # Simulate original behavior with freezing model
            # While freezing we do a cutting of model, to keep similar behavior we
            # need to simulate similar behavior with natively supported model
            outputs = fe_output_user_data_repack(input_model, argv.output, moc_front_end.get_name())
            input_model.override_all_outputs([x['node'] for x in outputs])
    '''

    enabled_transforms, disabled_transforms = get_enabled_and_disabled_transforms()
    if 'ANALYSIS_JSON_PRINT' in enabled_transforms:
        # NOTE that model analysis is performed before applying user's settings (inputs's shapes etc.)
        framework_model = moc_front_end.decode(input_model)
        json_model_analysis_dump(framework_model)
        # a model is not processed further in json analysis mode
        sys.exit(0)

    def check_places_are_same(places_original: List[Place], places_new: List[Place]):
        """
        Check if set of new places is same as original or not.
        :param places_original: List[Place] Original model places
        :param places_new: List[Place] New list of places
        :return: True if new list of places is same as original
        """
        return len(places_original) == len(places_new) and len(
            [item for item in places_original if any(
                [item.is_equal(item2['node']) for item2 in places_new])]) == len(places_original)

    if getattr(argv, "framework", None) == "pytorch":
        iplaces = []
        for idx, input_info in enumerate(argv.input):
            if getattr(input_info, "name", None):
                place = input_model.get_place_by_tensor_name(input_info.name)
                if not input_info.shape and not input_info.type:
                    # If we received place by name, we need to use it for FE to verify
                    # that such name exist, otherwise we silently ignore it.
                    # Using dynamic shape should be safe, because FE will not overwrite
                    # the shape that was produced after conversion, but merge it, so
                    # dynamic shape will not change anything.
                    input_model.set_partial_shape(place, PartialShape.dynamic())
            else:
                place = input_model.get_place_by_input_index(idx)
            iplaces.append(place)
            if input_info.shape is not None:
                input_model.set_partial_shape(place, input_info.shape)
            if input_info.type is not None:
                input_model.set_element_type(place, input_info.type)
        model_inputs = input_model.get_inputs()
        def merge_inputs(inputs, to_set_list):
            # use input places instead of obtained by index if they are the same
            res = []
            for p in to_set_list:
                found = False
                for i in inputs:
                    if p.is_equal(i):
                        res.append(i)
                        found = True
                        break
                if not found:
                    res.append(p)
            return res
        iplaces = merge_inputs(model_inputs, iplaces)
        oplaces = []
        # Currently this only work to reorder inputs
        to_override_all_inputs = check_places_are_same(model_inputs, [{"node": p} for p in iplaces])
        to_override_all_outputs = False
        if argv.output:
            _outputs = fe_output_user_data_repack(input_model, argv.output, moc_front_end.get_name())
            assert len(_outputs) == 0, "`output` argument is not supported for PyTorch"
        if to_override_all_inputs and to_override_all_outputs:
            input_model.extract_subgraph(iplaces, oplaces)
        elif to_override_all_inputs:
            input_model.override_all_inputs(iplaces)
        elif to_override_all_outputs:
            input_model.override_all_outputs(oplaces)

        ov_model = moc_front_end.convert(input_model)
        return ov_model

    argv.placeholder_shapes, argv.placeholder_data_types = convert_params_lists_to_dicts(
        input_model, argv.placeholder_shapes, argv.placeholder_data_types)

    user_shapes, outputs, freeze_placeholder = fe_user_data_repack(
        input_model, argv.placeholder_shapes, argv.placeholder_data_types,
        argv.output, {}, moc_front_end.get_name())

    def add_names_to_tensors(model: InputModel, places: List[Place]):
        """
        Adds additional names to some model input tensors. This helper should be used
        when a model modification is going to happen.
        :param model The input model loaded by a given frontend
        :param places An object containing Places and names that will be used for model modification
        """
        for new_input in places:
            if 'input_name' not in new_input:
                continue
            try:
                model.add_name_for_tensor(new_input['node'], new_input['input_name'])
            except NotImplementedFailure as e:
                # some frontends might not implement this method
                log.warning('Could not add an additional name to a tensor pointed to by \'{}\'. Details: {}'.format(
                    new_input['input_name'], str(e)))

    model_inputs = input_model.get_inputs()
    inputs_equal = True
    if user_shapes:
        # TODO: Remove this line when new 'cut' helper is introduced
        raise_exception_for_input_output_cut(model_inputs, user_shapes, True)

        inputs_equal = check_places_are_same(model_inputs, user_shapes)

    outputs_equal = True
    if outputs:
        # TODO: Remove this line when new 'cut' helper is introduced
        raise_exception_for_input_output_cut(input_model.get_outputs(), outputs, False)

        outputs_equal = check_places_are_same(input_model.get_outputs(), outputs)
    log.debug('Inputs are same: {}, outputs are same: {}'.format(
        inputs_equal, outputs_equal))

    def create_target_input_shapes(new_input_places):
        if isinstance(new_input_places, list) and len(new_input_places) > 1 \
                and isinstance(new_input_places[0], tuple):
            return new_input_places
        new_input_place_names = [x.get_names()[0] for x in new_input_places]
        shapes = [shape for shape in argv.placeholder_shapes.values()]
        return dict(zip(new_input_place_names, shapes))

    if not inputs_equal and not outputs_equal:
        log.debug('Using extract subgraph')
        new_input_places = [x['node'] for x in user_shapes]
        new_output_places = [x['node'] for x in outputs]
        add_names_to_tensors(input_model, user_shapes)
        input_model.extract_subgraph(new_input_places, new_output_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            placeholder_shapes = create_target_input_shapes(new_input_places)
            new_output_places_name = [x.get_names()[0] for x in new_output_places]

            user_shapes, outputs, _ = fe_user_data_repack(
                input_model, placeholder_shapes, argv.placeholder_data_types,
                new_output_places_name, {}, moc_front_end.get_name())
    elif not inputs_equal:
        log.debug('Using override_all_inputs')
        add_names_to_tensors(input_model, user_shapes)
        new_input_places = [x['node'] for x in user_shapes]
        input_model.override_all_inputs(new_input_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            placeholder_shapes = create_target_input_shapes(new_input_places)

            user_shapes, outputs, _ = fe_user_data_repack(
                input_model, placeholder_shapes, argv.placeholder_data_types,
                argv.output, {}, moc_front_end.get_name())
    elif not outputs_equal:
        log.debug('Using override_all_outputs')
        add_names_to_tensors(input_model, user_shapes)
        new_output_places = [x['node'] for x in outputs]
        input_model.override_all_outputs(new_output_places)
        # invalidation of existing Place objects could have happened in the operation above
        if user_shapes:
            model_inputs = input_model.get_inputs()

    if user_shapes:
        for user_shape in user_shapes:
            if user_shape.get('shape') is not None:
                input_model.set_partial_shape(
                    user_shape['node'], user_shape['shape'])
            if user_shape.get('data_type') is not None:
                data_type = user_shape['data_type']
                log.debug('Set data type: {}'.format(data_type))
                input_model.set_element_type(user_shape['node'], data_type)

    if freeze_placeholder:
        for name, value in freeze_placeholder.items():
            node = None
            # look for the certain place in user_shapes
            for node_cur in user_shapes:
                if node_cur.get('input_name') == name:
                    node = node_cur
                    break
            if node is None:
                raise Error("Please check correctness of the command-line. "
                            "Place (operation or tensor) with name {} is not found.".format(name))
            place = node.get('node')

            if node.get('data_type'):
                dtype = node['data_type']
                ov_type = Type(dtype)
            else:
                # we need to detect type of Placeholder
                try:
                    ov_type = input_model.get_element_type(place)
                except NotImplementedFailure:
                    raise Error("Please specify type for value freezing {} node explicitly "
                                "because the frontend does not support automatic type detection.".format(name))
                # in case of cutting graph (or using custom inputs) and unspecified or dynamic type,
                # the default type is fp32
                if ov_type == Type.undefined or ov_type == Type.dynamic:
                    ov_type = Type.f32
                dtype = get_numpy_ctype(ov_type)

            input_model.set_element_type(place, ov_type)
            # prepare and cast value to dtype
            if isinstance(value, list):
                casted_list = list()
                for v in mo_array(value):
                    casted_list.append(np_map_cast[dtype](v))
                value = mo_array(casted_list, dtype=dtype)
            else:
                value = np_map_cast[dtype](value)
            value = np.array(value, dtype=dtype)

            ov_shape = input_model.get_partial_shape(place)
            if node.get('shape'):
                # set user defined shape
                ov_shape = PartialShape(node['shape'])
                input_model.set_partial_shape(place, ov_shape)
            elif ov_shape.is_dynamic:
                # in case of dynamic shape (dynamic rank or dynamic dimension)
                # deduce it based on the value shape and set it
                ov_shape = PartialShape(value.shape)
                input_model.set_partial_shape(place, ov_shape)

            input_model.set_tensor_value(place, value)

    ov_model = moc_front_end.convert(input_model)

    return ov_model
