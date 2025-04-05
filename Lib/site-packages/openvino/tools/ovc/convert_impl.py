# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging as log
import os
import sys
import traceback
import tracemalloc
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Callable

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm

from openvino.tools.ovc.moc_frontend.check_config import any_extensions_used
from openvino.tools.ovc.moc_frontend.pipeline import moc_pipeline
from openvino.tools.ovc.moc_frontend.moc_emit_ir import moc_emit_ir
from openvino.tools.ovc.moc_frontend.type_utils import to_ov_type
from openvino.tools.ovc.cli_parser import get_available_front_ends, get_common_cli_options, depersonalize, \
    get_mo_convert_params, input_to_input_cut_info, parse_inputs
from openvino.tools.ovc.help import get_convert_model_help_specifics

from openvino.tools.ovc.error import Error, FrameworkError
from openvino.tools.ovc.get_ov_update_message import get_compression_message
from openvino.tools.ovc.version import VersionChecker
from openvino.tools.ovc.utils import check_values_equal
from openvino.tools.ovc.logger import init_logger
from openvino.tools.ovc.telemetry_utils import send_params_info, send_conversion_result, \
    init_ovc_telemetry
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import get_pytorch_decoder, \
    extract_input_info_from_example, get_pytorch_decoder_for_model_on_disk
from openvino.tools.ovc.moc_frontend.paddle_frontend_utils import paddle_frontend_converter

try:
    from openvino.tools.ovc.moc_frontend.jax_frontend_utils import get_jax_decoder
except:
    get_jax_decoder = None

# pylint: disable=no-name-in-module,import-error
from openvino.frontend import FrontEndManager, OpConversionFailure, TelemetryExtension
from openvino import get_version as get_rt_version
from openvino import PartialShape

try:
    from openvino.frontend.tensorflow.utils import create_tf_graph_iterator, type_supported_by_tf_fe, \
        extract_model_graph  # pylint: disable=no-name-in-module,import-error

    tf_frontend_with_python_bindings_installed = True
except (ModuleNotFoundError, ImportError):
    tf_frontend_with_python_bindings_installed = False


def replace_ext(name: str, old: str, new: str):
    base, ext = os.path.splitext(name)
    log.debug("base: {}, ext: {}".format(base, ext))
    if ext == old:
        return base + new


def print_argv(argv: argparse.Namespace):
    print('Model Conversion arguments:')
    props = OrderedDict()
    props['common_args'] = get_common_cli_options(argv, argv.is_python_api_used)

    framework_specifics_map = {
        'common_args': 'Common parameters:'
    }

    lines = []
    for key in props:
        lines.append(framework_specifics_map[key])
        for (op, desc) in props[key].items():
            if isinstance(desc, list):
                lines.append('\t{}: \t{}'.format(desc[0], desc[1](getattr(argv, op, 'NONE'))))
            else:
                lines.append('\t{}: \t{}'.format(desc, getattr(argv, op, 'NONE')))
    print('\n'.join(lines), flush=True)


def check_iterable(iterable: Iterable, func: Callable):
    for element in iterable:
        if not func(element):
            return False
    return True


def arguments_post_parsing(argv: argparse.Namespace):
    # TODO: This function looks similar to another one. Check for code duplicates.
    log.debug("Model Conversion API started")
    if not argv.is_python_api_used:
        log.debug('Output model name would be {}{{.xml, .bin}}'.format(argv.output_model))

    if is_verbose(argv):
        print_argv(argv)

    import re
    if argv.is_python_api_used and isinstance(argv.input, str):
        argv.input = [argv.input]

    if not argv.is_python_api_used and isinstance(argv.input, str):
        argv.input = parse_inputs(argv.input)

    normalize_inputs(argv)
    log.debug("Placeholder shapes : {}".format(argv.placeholder_shapes))

    if not hasattr(argv, 'output') or argv.output is None:
        return argv

    if argv.is_python_api_used:
        error_msg = f"output '{argv.output}' is incorrect, it should be string or a list/tuple of strings"
        assert isinstance(argv.output, (str, list, tuple)), error_msg
        if isinstance(argv.output, list):
            assert check_iterable(argv.output, lambda x: isinstance(x, str)), error_msg
        else:
            argv.output = [argv.output]
    else:
        assert isinstance(argv.output, str)

        error_msg = f"output '{argv.output}' is incorrect, output names should not be empty or contain spaces"
        processed_output = re.split(r'\s*,\s*', argv.output.strip())
        assert check_iterable(processed_output, lambda x: x.find(' ') == -1), error_msg
        assert check_iterable(processed_output, lambda x: len(x) > 0), error_msg
        argv.output = processed_output
    return argv


def get_moc_frontends(argv: argparse.Namespace):
    fem = argv.feManager

    if not fem:
        return None, []

    available_moc_front_ends = get_available_front_ends(fem)
    if argv.framework:
        moc_front_end = fem.load_by_framework(argv.framework)
        return moc_front_end, available_moc_front_ends
    if argv.input_model:
        if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 2:
            moc_front_end = fem.load_by_model(
                [argv.input_model[0], argv.input_model[1]])  # TODO: Pass all input model parts
        else:
            moc_front_end = fem.load_by_model(argv.input_model)
        if not moc_front_end:
            return None, available_moc_front_ends
        argv.framework = moc_front_end.get_name()
    else:
        return None, []

    # This check as a workaround to skip IR frontend
    if not moc_front_end.get_name() in available_moc_front_ends:
        return None, available_moc_front_ends

    return moc_front_end, available_moc_front_ends


def filtered_extensions(extensions):
    try:
        new_extensions = []
        from openvino.frontend.pytorch.module_extension import ModuleExtension
        for ext in extensions:
            if not isinstance(ext, ModuleExtension):
                new_extensions.append(ext)
        return new_extensions
    except:
        return extensions


def prepare_ir(argv: argparse.Namespace):
    argv = arguments_post_parsing(argv)
    t = tm.Telemetry()

    if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 1:
        argv.input_model = argv.input_model[0]

    moc_front_end, available_moc_front_ends = get_moc_frontends(argv)
    if moc_front_end:
        # TODO: Should be moved to the same place where paddle and pytorch handle their objects
        if argv.framework == 'tf' and argv.is_python_object and type_supported_by_tf_fe(argv.input_model):
            argv.input_model = create_tf_graph_iterator(argv.input_model,
                                                        argv.placeholder_shapes,
                                                        argv.placeholder_data_types,
                                                        getattr(argv, "example_input", None),
                                                        argv.share_weights)
        t.send_event("ovc", "conversion_method", moc_front_end.get_name() + "_frontend")
        moc_front_end.add_extension(TelemetryExtension("ovc", t.send_event, t.send_error, t.send_stack_trace))
        if any_extensions_used(argv):
            for extension in filtered_extensions(argv.extension):
                moc_front_end.add_extension(extension)
        ov_model = moc_pipeline(argv, moc_front_end)
        return ov_model

    if not argv.input_model:
        raise Error('No input model is provided')

    raise Error('Cannot recognize input model.')


def check_model_object(argv):
    model = argv['input_model']
    if 'tensorflow' in sys.modules:
        if tf_frontend_with_python_bindings_installed and extract_model_graph(argv):
            return "tf"
    if 'torch' in sys.modules:
        import torch
        if isinstance(model, (torch.nn.Module, torch.jit.ScriptFunction)) or (hasattr(torch, "export") and isinstance(model, (torch.export.ExportedProgram))):
            return "pytorch"
        try:
            from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
            from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder

            if isinstance(model, (TorchScriptPythonDecoder, TorchFXPythonDecoder)):
                return "pytorch"
        except Exception as e:
            pass

    import io
    # FIXME: Consuming any io.BytesIO object as an ONNX model is too dengerous and
    # can conflict with others in the future (not future proof).
    # TODO: Refer to https://onnx.ai/onnx/intro/python.html to find examples with
    # real ONNX python objects. ONNX model has onnx.onnx_ml_pb2.ModelProto type.
    if isinstance(model, io.BytesIO):
        return 'onnx'

    if 'paddle' in sys.modules:
        import paddle
        if isinstance(model, paddle.hapi.model.Model) or isinstance(model,
                                                                    paddle.fluid.dygraph.layers.Layer) or isinstance(
            model, paddle.fluid.executor.Executor):
            return "paddle"

    if 'jax' in sys.modules:
        import jax
        if isinstance(model, (jax.core.Jaxpr, jax.core.ClosedJaxpr)):
            return "jax"

    raise Error('Unknown model type: {}'.format(type(model)))


def driver(argv: argparse.Namespace, non_default_params: dict):
    # Log dictionary with non-default cli parameters where complex classes are excluded.
    log.debug(str(non_default_params))

    ov_model = moc_emit_ir(prepare_ir(argv), argv)

    return ov_model


def get_non_default_params(argv, cli_parser):
    import numbers
    import inspect
    from openvino.tools.ovc import convert_model

    signature = inspect.signature(convert_model)
    # make dictionary with parameters which have non-default values to be serialized in IR in rt_info
    non_default_params = {}
    for arg, arg_value in vars(argv).items():
        if arg in signature.parameters and check_values_equal(arg_value, signature.parameters[arg].default):
            continue
        if check_values_equal(arg_value, cli_parser.get_default(arg)):
            continue
        value = depersonalize(arg_value, arg)
        # Skip complex classes in params to prevent
        # serializing it to rt_info
        if isinstance(value, (str, bool, numbers.Number)):
            non_default_params[arg] = value
    return non_default_params


def add_line_breaks(text: str, char_num: int, line_break: str):
    words = text.replace('\n', "\n ").split(" ")
    cnt = 0
    for i, w in enumerate(words):
        cnt += len(w)
        if '\n' in w:
            cnt = len(w) - w.find('\n') - 1
        if cnt > char_num:
            if words[i][-1] not in ['\n', '\t']:
                words[i] = w + '\n'
            cnt = 0
    text = ' '.join(words).replace("\n ", "\n")
    return line_break + text.replace("\n", line_break)


def show_mo_convert_help():
    mo_convert_params = get_mo_convert_params()
    for group_name, group in mo_convert_params.items():
        print(group_name)
        for param_name in group:
            param_data = group[param_name]
            text = param_data.description.replace("    ", '')
            text = add_line_breaks(text, 56, "\n\t\t\t")
            print("  :param {} {}".format(param_name, text))
        print()


def input_model_is_object(input_model):
    if input_model == ():
        return False
    if isinstance(input_model, (str, Path)):
        return False
    if isinstance(input_model, (tuple, list)):
        return all(input_model_is_object(part) for part in input_model)
    return True


def normalize_inputs(argv: argparse.Namespace):
    """
    repacks params passed to convert_model and wraps resulting values into dictionaries or lists.
    After working of this method following values are set in argv:

    argv.input, argv.inputs_list - list of input names. Both values are used in some parts of MO.
    Could be good to refactor it and use only one of these values.

    argv.placeholder_shapes - dictionary where key is node name, value is PartialShape,
    or list of PartialShape if node names were not set.

    argv.placeholder_data_types - dictionary where key is node name, value is node np.type,
    or list of np.types if node names were not set.

    :param argv: OVC arguments
    """
    # Parse input to list of InputCutInfo
    inputs = input_to_input_cut_info(argv.input)
    argv.input = inputs

    # Make list of input names
    input_names_list = []
    for inp in inputs:
        if inp.name is not None:
            input_names_list.append(inp.name)
    if len(input_names_list) > 0:
        assert len(input_names_list) == len(inputs), "\"input\" parameter has unnamed inputs and named inputs. " \
                                                     "Please either set names for all inputs, " \
                                                     "or do not set names for all inputs."

    if len(input_names_list) > 0:
        # Named inputs case
        shape_dict = {}
        data_type_dict = {}
        for inp in inputs:
            if inp.shape is not None:
                # Wrap shape to PartialShape for uniformity of stored values
                shape_dict[inp.name] = PartialShape(inp.shape)
            else:
                shape_dict[inp.name] = None
            if inp.type is not None:
                # Convert type to ov.Type for uniformity of stored values
                data_type_dict[inp.name] = to_ov_type(inp.type)
        argv.placeholder_shapes = shape_dict if shape_dict else None
        argv.placeholder_data_types = data_type_dict if data_type_dict else {}
    else:
        # Unnamed inputs case
        shape_list = []
        data_type_list = []
        for inp in inputs:
            if inp.shape is not None:
                # Wrap shape to PartialShape for uniformity of stored values
                shape_list.append(PartialShape(inp.shape))
            if inp.type is not None:
                # Convert type to ov.Type for uniformity of stored values
                data_type_list.append(to_ov_type(inp.type))
        argv.placeholder_shapes = shape_list if shape_list else None
        argv.placeholder_data_types = data_type_list if data_type_list else {}
    if hasattr(argv, "framework") and argv.framework == "pytorch" and getattr(argv, "example_input", None) is not None:
        extract_input_info_from_example(argv, inputs)


def args_to_argv(**kwargs):
    argv = argparse.Namespace()
    args_specifics = get_convert_model_help_specifics()

    import inspect
    from openvino.tools.ovc import convert_model
    signature = inspect.signature(convert_model)
    for key, value in kwargs.items():
        if value is None and key in signature.parameters:
            setattr(argv, key, signature.parameters[key].default)
            continue
        if key in args_specifics:
            param_specifics = args_specifics[key]
            if 'action' in param_specifics and hasattr(param_specifics['action'], 'check_value'):
                value = param_specifics['action'].check_value(value, key)
            if 'type' in param_specifics:
                value = param_specifics['type'](value)
        setattr(argv, key, value)
    return argv


def pack_params_to_args_namespace(args: dict, cli_parser: argparse.ArgumentParser, python_api_used):
    if python_api_used:
        argv = args_to_argv(**args)

        # get list of all available params for convert_model()
        all_params = {}
        for key, value in get_mo_convert_params().items():
            all_params.update(value)

        # check that there are no unknown params provided
        for key, value in args.items():
            if key not in all_params.keys():
                raise Error("Unrecognized argument: {}".format(key))
    else:
        argv = cli_parser.parse_args()
    return argv


def is_verbose(argv, args=None):
    if argv is not None and hasattr(argv, 'verbose') and argv.verbose:
        return True
    if args is not None and 'verbose' in args and args['verbose']:
        return True
    if '--verbose' in sys.argv:
        return True
    return False


def _convert(cli_parser: argparse.ArgumentParser, args, python_api_used):
    start_time = datetime.datetime.now()
    if is_verbose(None, args):
        tracemalloc.start()

    simplified_ie_version = VersionChecker().get_ie_simplified_version()
    telemetry = init_ovc_telemetry()
    telemetry.start_session('ovc')
    telemetry.send_event('ovc', 'version', simplified_ie_version)
    # Initialize logger with 'ERROR' as default level to be able to form nice messages
    # before arg parser deliver log_level requested by user
    verbose = False
    if "verbose" in args and args["verbose"] or "--verbose" in sys.argv:
        verbose = True

    init_logger('ERROR', verbose, python_api_used)
    argv = None
    # Minimize modifications among other places in case if multiple pieces are passed as input_model
    if python_api_used:
        if 'input_model' not in args:
            args['input_model'] = ()
        if isinstance(args['input_model'], (tuple, list)) and len(args['input_model']) == 1:
            args['input_model'] = args['input_model'][0]
    try:
        model_framework = None
        inp_model_is_object = input_model_is_object(args['input_model']) if python_api_used else False

        if inp_model_is_object:
            model_framework = check_model_object(args)
            if model_framework == "pytorch":
                example_inputs = None
                if 'example_input' in args and args['example_input'] is not None:
                    example_inputs = args['example_input']
                elif 'example_inputs' in args:
                    raise AssertionError(
                        "'example_inputs' argument is not recognized, maybe you meant to provide 'example_input'?")

                get_pytorch_decoder(args['input_model'], example_inputs, args)
            if model_framework == "paddle":
                example_inputs = None
                if 'example_input' in args and args['example_input'] is not None:
                    example_inputs = args['example_input']

                outputs = None
                if 'output' in args and args['output'] is not None:
                    # Once the temporary PDPD model is generated. output can be dropped.
                    # Just swap outputs and args['output'] can reset the argv.output to `None`.
                    # It can avoid the following `output` negative effect.
                    outputs, args['output'] = args['output'], outputs
                paddle_runtime_converter = paddle_frontend_converter(args['input_model'], example_inputs,
                                                                     outputs)
                pdmodel = paddle_runtime_converter.convert_paddle_to_pdmodel()
                args['input_model'] = pdmodel
            if model_framework == "jax":
                if get_jax_decoder is not None:
                    get_jax_decoder(args['input_model'], args)
                else:
                    raise Error("JAX Frontend is not available.")

        argv = pack_params_to_args_namespace(args, cli_parser, python_api_used)

        argv.framework = model_framework
        argv.is_python_object = inp_model_is_object

        argv.feManager = FrontEndManager()

        non_default_params = get_non_default_params(argv, cli_parser)
        argv.is_python_api_used = python_api_used

        # send telemetry with params info
        send_params_info(non_default_params)

        argv.framework = model_framework

        orig_input_model = argv.input_model
        pytorch_model_on_disk = False
        if argv.framework is None and get_pytorch_decoder_for_model_on_disk(argv, args):
            # try to load a model from disk as TorchScript or ExportedProgram
            # TorchScriptPythonDecoder or TorchFXPythonDecoder object will be assigned to argv.input_model
            # saved TorchScript and ExportedModel model can be passed to both ovc tool and Python convert_model
            pytorch_model_on_disk = True

        ov_model = driver(argv, {"conversion_parameters": non_default_params})

        if pytorch_model_on_disk:
            # release memory allocated for temporal object
            del argv.input_model
            # restore original model name in arguments for tool reporting
            argv.input_model = orig_input_model

        if inp_model_is_object and model_framework == "paddle":
            if paddle_runtime_converter:
                paddle_runtime_converter.destroy()

        # add OVC meta data to model
        ov_model.set_rt_info(get_rt_version(), "Runtime_version")
        for key, value in non_default_params.items():
            ov_model.set_rt_info(str(value), ["conversion_parameters", str(key)])

        if is_verbose(argv) or not python_api_used:
            if 'compress_to_fp16' in argv and argv.compress_to_fp16:
                print(get_compression_message())

        send_conversion_result('success')

        if is_verbose(argv):
            elapsed_time = datetime.datetime.now() - start_time
            print('[ SUCCESS ] Total execution time: {:.2f} seconds. '.format(elapsed_time.total_seconds()))

            _, peak_size = tracemalloc.get_traced_memory()
            print("[ SUCCESS ] Peak memory consumption (includes only memory allocated in Python): {:.2f} MB. ".format(
                peak_size / (1024 * 1024)))
            tracemalloc.stop()

        return ov_model, argv

    except Exception as e:
        if is_verbose(argv) or not python_api_used:
            if isinstance(e, (FileNotFoundError, NotADirectoryError)):
                log.error('File {} was not found'.format(str(e).split('No such file or directory:')[1]))
                log.debug(traceback.format_exc())
            elif isinstance(e, (Error, OpConversionFailure)):
                log.error(e)
                log.debug(traceback.format_exc())
            elif isinstance(e, FrameworkError):
                log.error(e, extra={'framework_error': True})
                log.debug(traceback.format_exc())
            else:
                log.error("-------------------------------------------------")
                log.error("----------------- INTERNAL ERROR ----------------")
                log.error("Unexpected exception happened.")
                log.error("Please verify parameters and environment.")
                log.error("If you think this is a bug, please create new ticket here: ")
                log.error("https://github.com/openvinotoolkit/openvino/issues.")
                log.error("-------------- DETAILED INFORMATION -------------")
                log.error(str(e))
                log.error(traceback.format_exc())
                log.error("----------------- END OF REPORT -----------------")
                log.error("-------------------------------------------------")

        send_conversion_result('fail')
        if python_api_used:
            raise e
        else:
            return None, argv
