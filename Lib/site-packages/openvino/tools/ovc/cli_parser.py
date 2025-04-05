# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import inspect
import os
import pathlib
import re
from collections import OrderedDict, namedtuple
from typing import List, Union

import openvino
from openvino import PartialShape, Dimension, Type  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.help import get_convert_model_help_specifics
from openvino.tools.ovc.moc_frontend.shape_utils import to_partial_shape, is_shape_type
from openvino.tools.ovc.moc_frontend.type_utils import to_ov_type, is_type
from openvino.tools.ovc.utils import get_mo_root_dir

# Helper class for storing input cut information
_InputCutInfo = namedtuple("InputCutInfo", ["name", "shape", "type", "value"], defaults=[None, None, None, None])


def single_input_to_input_cut_info(input: [str, tuple, list, PartialShape, Type, type]):
    """
    Parses parameters of single input to InputCutInfo.
    :param input: input cut parameters of single input
    :return: InputCutInfo
    """
    if isinstance(input, str):
        # pylint: disable=no-member
        return _InputCutInfo(input, None)
    if isinstance(input, (tuple, list)) or is_shape_type(input):
        # If input represents list with shape, wrap it to list. Single PartialShape also goes to this condition.
        # Check of all dimensions will be in is_shape_type(val) method below
        if is_shape_type(input):
            input = [input]

        # Check values of tuple or list and collect to InputCutInfo
        name = None
        inp_type = None
        shape = None
        for val in input:
            if isinstance(val, str):
                if name is not None:
                    raise Exception("More than one input name provided: {}".format(input))
                name = val
            elif is_type(val):
                if inp_type is not None:
                    raise Exception("More than one input type provided: {}".format(input))
                inp_type = to_ov_type(val)
            elif is_shape_type(val) or val is None:
                if shape is not None:
                    raise Exception("More than one input shape provided: {}".format(input))
                shape = to_partial_shape(val) if val is not None else None
            else:
                raise Exception("Incorrect input parameters provided. Expected tuple with input name, "
                                "input type or input shape. Got unknown object: {}".format(val))
        # pylint: disable=no-member
        return _InputCutInfo(name,
                             PartialShape(shape) if shape is not None else None,
                             inp_type,
                             None)
    # Case when only type is set
    if is_type(input):
        return _InputCutInfo(None, None, to_ov_type(input), None)  # pylint: disable=no-member

    # We don't expect here single unnamed value. If list of int is set it is considered as shape.
    # Setting of value is expected only using InputCutInfo or string analog.

    raise Exception(
        "Unexpected object provided for input. Expected tuple, Shape, PartialShape, Type or str. Got {}".format(
            type(input)))


def is_single_input(input: [tuple, list]):
    """
    Checks if input has parameters for single input.
    :param input: list or tuple of input parameters or input shape or input name.
    :return: True if input has parameters for single input, otherwise False.
    """
    name = None
    inp_type = None
    shape = None
    for val in input:
        if isinstance(val, str):
            if name is not None:
                return False
            name = val
        elif is_type(val):
            if inp_type is not None:
                return False
            inp_type = to_ov_type(val)
        elif is_shape_type(val):
            if shape is not None:
                return False
            shape = to_partial_shape(val)
        else:
            return False
    return True


def parse_inputs(inputs: str):
    inputs_list = []
    # Split to list of string
    for input_value in split_inputs(inputs):
        # Parse string with parameters for single input
        node_name, shape = parse_input_value(input_value)
        # pylint: disable=no-member
        inputs_list.append((node_name, shape))
    return inputs_list


def input_to_input_cut_info(input: [dict, tuple, list]):
    """
    Parses 'input' to list of InputCutInfo.
    :param input: input cut parameters passed by user
    :return: list of InputCutInfo with input cut parameters
    """
    if input is None:
        return []

    if isinstance(input, (tuple, list)):
        if len(input) == 0:
            return []
        # Case when input is single shape set in tuple
        if len(input) > 0 and isinstance(input[0], (int, Dimension)):
            input = [input]

        if is_single_input(input):
            return [single_input_to_input_cut_info(input)]

        inputs = []
        for inp in input:
            inputs.append(single_input_to_input_cut_info(inp))
        return inputs

    if isinstance(input, dict):
        res_list = []
        for name, value in input.items():
            if not isinstance(name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(name)))
            info = single_input_to_input_cut_info(value)
            if info.name is not None and info.name != name:
                raise Exception("Incorrect \"input\" dictionary, got different names in key and value. "
                                "Got operation name {} for key {}".format(info.name, name))
            res_list.append(_InputCutInfo(name, info.shape, info.type))
        return res_list
    # Case when single type or value is set, or unknown object
    return [single_input_to_input_cut_info(input)]


ParamDescription = namedtuple("ParamData", ["description", "cli_tool_description"])


def get_mo_convert_params():
    mo_convert_docs = openvino.tools.ovc.convert_model.__doc__  # pylint: disable=no-member
    mo_convert_params = {}
    group = "Optional parameters:"  # FIXME: WA for unknown bug in this function
    mo_convert_params[group] = {}

    mo_convert_docs = mo_convert_docs[:mo_convert_docs.find('Returns:')]

    while len(mo_convert_docs) > 0:
        param_idx1 = mo_convert_docs.find(":param")
        if param_idx1 == -1:
            break
        param_idx2 = mo_convert_docs.find(":", param_idx1 + 1)
        param_name = mo_convert_docs[param_idx1 + len(':param '):param_idx2]

        param_description_idx = mo_convert_docs.find(":param", param_idx2 + 1)
        param_description = mo_convert_docs[param_idx2 + 1: param_description_idx]

        group_name_idx = param_description.rfind('\n\n')
        group_name = ''
        if group_name_idx != -1:
            group_name = param_description[group_name_idx:].strip()

        param_description = param_description[:group_name_idx]
        param_description = param_description.strip()

        mo_convert_params[group][param_name] = ParamDescription(param_description, "")

        mo_convert_docs = mo_convert_docs[param_description_idx:]

        if group_name != '':
            mo_convert_params[group_name] = {}
            group = group_name

    cli_tool_specific_descriptions = get_convert_model_help_specifics()

    for group_name, param_group in mo_convert_params.items():
        for param_name, d in param_group.items():
            cli_tool_description = None
            if param_name in cli_tool_specific_descriptions:
                cli_tool_description = cli_tool_specific_descriptions[param_name]

            desc = ParamDescription(d.description,
                                    cli_tool_description)
            mo_convert_params[group_name][param_name] = desc

    return mo_convert_params


def canonicalize_and_check_paths(values: Union[str, List[str], None], param_name,
                                 try_mo_root=False, check_existence=True) -> List[str]:
    if values is not None:
        list_of_values = list()
        if isinstance(values, str):
            if values != "":
                list_of_values = values.split(',')
        elif isinstance(values, list):
            list_of_values = values
        else:
            return values

        if not check_existence:
            return [get_absolute_path(path) for path in list_of_values]

        for idx, val in enumerate(list_of_values):
            if not isinstance(val, (str, pathlib.Path)):
                continue

            list_of_values[idx] = val

            error_msg = 'The value for parameter "{}" must be existing file/directory, ' \
                        'but "{}" does not exist.'.format(param_name, val)
            if os.path.exists(val):
                continue
            elif not try_mo_root or val == '':
                raise Error(error_msg)
            elif try_mo_root:
                path_from_mo_root = get_mo_root_dir() + '/ovc/' + val
                list_of_values[idx] = path_from_mo_root
                if not os.path.exists(path_from_mo_root):
                    raise Error(error_msg)

        return [get_absolute_path(path) for path in list_of_values]


class CanonicalizePathCheckExistenceAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
    existence.
    """
    check_value = canonicalize_and_check_paths

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existence=True)
        setattr(namespace, self.dest, list_of_paths)


def readable_file_or_dir_or_object(path: str):
    """
    Check that specified path is a readable file or directory.
    :param path: path to check
    :return: path if the file/directory is readable
    """
    if not isinstance(path, (str, pathlib.Path)):
        return path
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise Error('The "{}" is not existing file or directory'.format(path))
    elif not os.access(path, os.R_OK):
        raise Error('The "{}" is not readable'.format(path))
    else:
        return path


def readable_dirs_or_files_or_empty(paths: [str, list, tuple]):
    """
    Checks that comma separated list of paths are readable directories, files or a provided path is empty.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    paths_list = paths
    if isinstance(paths, (list, tuple)):
        paths_list = [readable_file_or_dir_or_object(path) for path in paths]
    if isinstance(paths, (str, pathlib.Path)):
        paths_list = [readable_file_or_dir_or_object(path) for path in str(paths).split(',')]

    return paths_list[0] if isinstance(paths, (list, tuple)) and len(paths_list) == 1 else paths_list


def add_args_by_description(args_group, params_description):
    signature = inspect.signature(openvino.tools.ovc.convert_model)  # pylint: disable=no-member
    filepath_args = get_params_with_paths_list()
    cli_tool_specific_descriptions = get_convert_model_help_specifics()
    for param_name, param_description in params_description.items():
        if param_name in ['share_weights', 'example_input']:
            continue
        if param_name == 'input_model':
            # input_model is not a normal key for a tool, it will collect all untagged keys
            cli_param_name = param_name
        else:
            cli_param_name = '--' + param_name
        if cli_param_name not in args_group._option_string_actions:
            # Get parameter specifics
            param_specifics = cli_tool_specific_descriptions[param_name] if param_name in \
                                                                            cli_tool_specific_descriptions else {}
            help_text = param_specifics['description'] if 'description' in param_specifics \
                else param_description.description
            action = param_specifics['action'] if 'action' in param_specifics else None
            param_type = param_specifics['type'] if 'type' in param_specifics else None
            param_alias = param_specifics[
                'aliases'] if 'aliases' in param_specifics and param_name != 'input_model' else {}
            param_version = param_specifics['version'] if 'version' in param_specifics else None
            param_choices = param_specifics['choices'] if 'choices' in param_specifics else None

            # Bool params common setting
            if signature.parameters[param_name].annotation == bool and param_name != 'version':
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    action='store_true',
                    help=help_text,
                    default=signature.parameters[param_name].default)
            # File paths common setting
            elif param_name in filepath_args:
                action = action if action is not None else CanonicalizePathCheckExistenceAction
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    type=str if param_type is None else param_type,
                    action=action,
                    help=help_text,
                    default=None if param_name == 'input_model' else signature.parameters[param_name].default,
                    metavar=param_name.upper() if param_name == 'input_model' else None)
            # Other params
            else:
                additional_params = {}
                if param_version is not None:
                    additional_params['version'] = param_version
                if param_type is not None:
                    additional_params['type'] = param_type
                if param_choices is not None:
                    additional_params['choices'] = param_choices
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    help=help_text,
                    default=signature.parameters[param_name].default,
                    action=action,
                    **additional_params
                )


class Formatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        usage = argparse.HelpFormatter._format_usage(self, usage, actions, groups, prefix)
        usage = usage[0:usage.find('INPUT_MODEL')].rstrip() + '\n'
        insert_idx = usage.find(self._prog) + len(self._prog)
        usage = usage[0: insert_idx] + ' INPUT_MODEL... ' + usage[insert_idx + 1:]
        return usage

    def _get_default_metavar_for_optional(self, action):
        if action.option_strings == ['--compress_to_fp16']:
            return "True | False"
        return argparse.HelpFormatter._get_default_metavar_for_optional(self, action)


def get_common_cli_parser(parser: argparse.ArgumentParser = None):
    if not parser:
        parser = argparse.ArgumentParser(formatter_class=Formatter)
    mo_convert_params = get_mo_convert_params()
    mo_convert_params_common = mo_convert_params['Optional parameters:']

    from openvino.tools.ovc.version import VersionChecker

    # Command line tool specific params
    parser.add_argument('--output_model',
                        help='This parameter is used to name output .xml/.bin files of converted model. '
                             'Model name or output directory can be passed. If output directory is passed, '
                             'the resulting .xml/.bin files are named by original model name.')
    parser.add_argument('--compress_to_fp16', type=check_bool, default=True, nargs='?',
                        help='Compress weights in output OpenVINO model to FP16. '
                             'To turn off compression use "--compress_to_fp16=False" command line parameter. '
                             'Default value is True.')
    parser.add_argument('--version', action='version',
                        help='Print ovc version and exit.',
                        version='OpenVINO Model Converter (ovc) {}'.format(VersionChecker().get_ie_version()))
    add_args_by_description(parser, mo_convert_params_common)
    return parser


def input_model_details(model):
    if isinstance(model, (list, tuple)) and len(model) == 1:
        model = model[0]
    if isinstance(model, (str, pathlib.Path)):
        return model
    return type(model)


def get_common_cli_options(argv, is_python_api_used):
    d = OrderedDict()
    d['input_model'] = ['- Input Model', input_model_details]
    if not is_python_api_used:
        model_name = get_model_name_from_args(argv)
        d['output_model'] = ['- IR output name', lambda _: model_name]
    d['input'] = ['- Input layers', lambda x: x if x else 'Not specified, inherited from the model']
    d['output'] = ['- Output layers', lambda x: x if x else 'Not specified, inherited from the model']
    return d


def get_params_with_paths_list():
    return ['input_model', 'output_model', 'extension']


def get_all_cli_parser():
    """
    Specifies cli arguments for Model Conversion

    Returns
    -------
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(formatter_class=Formatter)

    get_common_cli_parser(parser=parser)

    return parser


def remove_shape_from_input_value(input_value: str):
    """
    Removes the shape specification from the input string. The shape specification is a string enclosed with square
    brackets.
    :param input_value: string passed as input to the "input" command line parameter
    :return: string without shape specification
    """
    if '->' in input_value:
        raise Error('Incorrect format of input. Got {}'.format(input_value))
    return re.sub(r'[(\[]([0-9\.?,  -]*)[)\]]', '', input_value)


def get_shape_from_input_value(input_value: str):
    """
    Returns PartialShape corresponding to the shape specified in the input value string
    :param input_value: string passed as input to the "input" command line parameter
    :return: the corresponding shape and None if the shape is not specified in the input value
    """

    # parse shape
    shape = re.findall(r'[(\[]([0-9\.\?,  -]*)[)\]]', input_value)
    if len(shape) == 0:
        shape = None
    elif len(shape) == 1 and shape[0] in ['', ' ']:
        # this shape corresponds to scalar
        shape = PartialShape([])
    elif len(shape) == 1:
        dims = re.split(r', *| +', shape[0])
        dims = list(filter(None, dims))
        shape = PartialShape([Dimension(dim) for dim in dims])
    else:
        raise Error("Wrong syntax to specify shape. Use \"input\" "
                    "\"node_name[shape]\"")
    return shape


def get_node_name_with_port_from_input_value(input_value: str):
    """
    Returns the node name (optionally with input/output port) from the input value
    :param input_value: string passed as input to the "input" command line parameter
    :return: the corresponding node name with input/output port
    """
    return remove_shape_from_input_value(input_value)


def parse_input_value(input_value: str):
    """
    Parses a value of the "input" command line parameter and gets a node name, shape and value.
    The node name includes a port if it is specified.
    Shape and value is equal to None if they are not specified.
    Parameters
    ----------
    input_value
        string with a specified node name and shape.
        E.g. 'node_name:0[4]'

    Returns
    -------
        Node name, shape, value, data type
        E.g. 'node_name:0', '4', [1.0 2.0 3.0 4.0], np.float32
    """
    node_name = get_node_name_with_port_from_input_value(input_value)
    shape = get_shape_from_input_value(input_value)

    return node_name if node_name else None, shape


def split_inputs(input_str):
    pattern = r'^(?:[^[\]()<]*(\[[\.)-9,\-\s?]*\])*,)*[^[\]()<]*(\[[\.0-9,\-\s?]*\])*$'
    if not re.match(pattern, input_str):
        raise Error(f"input value '{input_str}' is incorrect. Input should be in the following format: "
                    f"{get_convert_model_help_specifics()['input']['description']}")

    brakets_count = 0
    inputs = []
    while input_str:
        idx = 0
        for c in input_str:
            if c == '[':
                brakets_count += 1
            if c == ']':
                brakets_count -= 1
            if c == ',':
                if brakets_count != 0:
                    idx += 1
                    continue
                else:
                    break
            idx += 1
        if idx >= len(input_str) - 1:
            inputs.append(input_str)
            break
        inputs.append(input_str[:idx])
        input_str = input_str[idx + 1:]
    return inputs


def get_model_name(path_input_model: str) -> str:
    """
    Deduces model name by a given path to the input model
    Args:
        path_input_model: path to the input model

    Returns:
        name of the output IR
    """
    parsed_name, extension = os.path.splitext(os.path.basename(path_input_model))
    return 'model' if parsed_name.startswith('.') or len(parsed_name) == 0 else parsed_name


def get_model_name_from_args(argv: argparse.Namespace):
    output_dir = os.getcwd()
    if hasattr(argv, 'output_model') and argv.output_model:
        model_name = argv.output_model

        if not os.path.isdir(argv.output_model) and not argv.output_model.endswith(os.sep):
            # In this branch we assume that model name is set in 'output_model'.
            if not model_name.endswith('.xml'):
                model_name += '.xml'
            # Logic of creating and checking directory is covered in save_model() method.
            return model_name
        else:
            # In this branch 'output_model' has directory without name of model.
            # The directory may not exist.
            if os.path.isdir(argv.output_model) and not os.access(argv.output_model, os.W_OK):
                # If the provided path is existing directory, but not writable, then raise error
                raise Error('The directory "{}" is not writable'.format(argv.output_model))
            output_dir = argv.output_model

    input_model = argv.input_model
    if isinstance(input_model, (tuple, list)) and len(input_model) > 0:
        input_model = input_model[0]

    input_model = os.path.abspath(input_model)

    if not isinstance(input_model, (str, pathlib.Path)):
        return output_dir

    input_model_name = os.path.basename(input_model)
    if input_model_name == '':
        input_model_name = os.path.basename(os.path.dirname(input_model))

    # remove extension if exists
    input_model_name = os.path.splitext(input_model_name)[0]

    # if no valid name exists in input path set name to 'model'
    if input_model_name == '':
        raise Exception("Could not derive model name from input model. Please provide 'output_model' parameter.")

    # add .xml extension
    return os.path.join(output_dir, input_model_name + ".xml")


def get_absolute_path(path_to_file: str) -> str:
    """
    Deduces absolute path of the file by a given path to the file
    Args:
        path_to_file: path to the file

    Returns:
        absolute path of the file
    """
    if not isinstance(path_to_file, (str, pathlib.Path)):
        return path_to_file
    file_path = os.path.expanduser(path_to_file)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path


def check_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() not in ['true', 'false']:
            raise argparse.ArgumentTypeError("expected a True/False value")
        return value.lower() == 'true'
    else:
        raise argparse.ArgumentTypeError("expected a bool or str type")


def depersonalize(value: str, key: str):
    dir_keys = [
        'extension'
    ]
    if isinstance(value, list):
        updated_value = []
        for elem in value:
            updated_value.append(depersonalize(elem, key))
        return updated_value

    if not isinstance(value, str):
        return value
    res = []
    for path in value.split(','):
        if os.path.isdir(path) and key in dir_keys:
            res.append('DIR')
        elif os.path.isfile(path):
            res.append(os.path.join('DIR', os.path.split(path)[1]))
        else:
            res.append(path)
    return ','.join(res)


def get_available_front_ends(fem=None):
    # Use this function as workaround to avoid IR frontend usage by OVC
    if fem is None:
        return []
    available_moc_front_ends = fem.get_available_front_ends()
    if 'ir' in available_moc_front_ends:
        available_moc_front_ends.remove('ir')

    return available_moc_front_ends
