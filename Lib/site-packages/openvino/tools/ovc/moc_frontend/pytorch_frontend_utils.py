# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import pathlib
import sys

import numpy as np

# pylint: disable=no-name-in-module,import-error
from openvino import Tensor, PartialShape
from openvino.tools.ovc.cli_parser import single_input_to_input_cut_info, _InputCutInfo
from openvino.tools.ovc.error import Error


def extract_module_extensions(args):
    from openvino.frontend.pytorch.module_extension import ModuleExtension
    extensions = args.get('extension', []) or []
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]
    return {extension.module: extension for extension in extensions if isinstance(extension, ModuleExtension)}


def get_decoder_for_exported_program(model):
    from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
    import torch

    from packaging import version
    if version.parse(torch.__version__) >= version.parse("2.2"):
        from torch._decomp import get_decompositions
        from openvino.frontend.pytorch.torchdynamo.decompositions import get_export_decomposition_list
        decomp = get_decompositions(get_export_decomposition_list())
        model = model.run_decompositions(decomp_table=decomp)
    gm = model.module()
    log.debug(gm.code)
    decoder = TorchFXPythonDecoder(gm, dynamic_shapes=True)
    return decoder


def get_pytorch_decoder(model, example_inputs, args):
    try:
        from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
        from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
        from openvino.frontend.pytorch.module_extension import ModuleExtension
        import torch
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e

    if 'nncf' in sys.modules:
        is_good_version = True
        try:
            from nncf.torch.nncf_network import NNCFNetwork

            if isinstance(model, NNCFNetwork):
                from packaging import version
                if version.parse(sys.modules['nncf'].__version__) < version.parse("2.6"):
                    is_good_version = False
        except:
            pass
        if not is_good_version:
            raise RuntimeError("NNCF models produced by nncf<2.6 are not "
                               "supported directly. Please upgrade nncf or "
                               "export to ONNX first.")
    inputs = prepare_torch_inputs(example_inputs)
    if not isinstance(model, (TorchScriptPythonDecoder, TorchFXPythonDecoder)):
        if hasattr(torch, "export") and isinstance(model, (torch.export.ExportedProgram)):
            decoder = get_decoder_for_exported_program(model)
        else:
            decoder = TorchScriptPythonDecoder(
                model,
                example_input=inputs,
                shared_memory=args.get("share_weights", True),
                module_extensions=extract_module_extensions(args))
    else:
        decoder = model
    args['input_model'] = decoder
    ei = getattr(decoder, "_example_input", None)
    if ei is not None:
        args["example_input"] = ei
    else:
        args["example_input"] = inputs

    return args


def get_pytorch_decoder_for_model_on_disk(argv, args):
    try:
        from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
        from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
        import torch
    except:
        return False

    example_inputs = None
    if 'example_input' in args and args['example_input'] is not None:
        example_inputs = args['example_input']

    if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 1:
        input_model = argv.input_model[0]
    else:
        input_model = argv.input_model

    if not isinstance(input_model, (str, pathlib.Path)):
        return False

    # attempt to load scripted model
    try:
        inputs = prepare_torch_inputs(example_inputs)
        model = torch.jit.load(input_model)
        model.eval()
        decoder = TorchScriptPythonDecoder(
            model,
            example_input=inputs,
            shared_memory=args.get("share_weights", True),
            module_extensions=extract_module_extensions(args))
        argv.input_model = decoder
        argv.framework = 'pytorch'
        return True
    except:
        pass
    # attempt to load exported model
    try:
        exported_program = torch.export.load(input_model)
        if hasattr(torch, "export") and isinstance(exported_program, (torch.export.ExportedProgram)):
            argv.input_model = get_decoder_for_exported_program(exported_program)
            argv.framework = 'pytorch'
            return True
    except:
        pass
    return False


def update_list_or_dict(container, name, idx, value):
    if isinstance(container, dict):
        if name is None:
            name = list(container)[idx]
        container[name] = value
        return
    if idx == len(container):
        container.append(value)
    elif idx > len(container):
        raise Error(f"Wrong {idx}")
    else:
        container[idx] = value
    return


def get_value_from_list_or_dict(container, name, idx):
    if isinstance(container, dict):
        if name is None:
            if idx < len(container):
                name = list(container)[idx]
            return None
        return container.get(name)
    if idx < len(container):
        return container[idx]
    return None


def flatten_inputs(inputs, names=None):
    flattened = []
    if isinstance(inputs, dict):
        # if names are provided we need to unpack in the same order
        if names:
            for name in names:
                if isinstance(inputs[name], (list, tuple, dict)):
                    flattened.extend(flatten_inputs(inputs[name]))
                else:
                    flattened.append((name, inputs[name]))
        else:
            for name, input_data in inputs.items():
                if isinstance(input_data, (list, tuple, dict)):
                    flattened.extend(flatten_inputs(input_data))
                else:
                    flattened.append((name, input_data))
    else:
        for input_data in inputs:
            if isinstance(input_data, (list, tuple, dict)):
                flattened.extend(flatten_inputs(input_data))
            else:
                flattened.append(input_data)
    return flattened


def extract_input_info_from_example(args, inputs):
    try:
        from openvino.frontend.pytorch.utils import pt_to_ov_type_map  # pylint: disable=no-name-in-module,import-error
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    example_inputs = args.input_model._example_input if args.input_model._example_input is not None else args.example_input
    if example_inputs is None:
        return
    is_dict_input = isinstance(example_inputs, dict)
    if not isinstance(example_inputs, (list, tuple, dict)):
        example_inputs = [example_inputs]
    input_names = None
    if args.input_model._input_signature is not None:
        input_names = args.input_model._input_signature[1:] if args.input_model._input_signature[
                                                                   0] == "self" else args.input_model._input_signature
    if input_names and not is_dict_input:
        example_inputs = dict(zip(input_names, example_inputs))
    example_inputs = flatten_inputs(example_inputs, input_names)
    input_arg = []
    for example in example_inputs:
        name = None
        if isinstance(example, tuple) and len(example) == 2:
            name = example[0]
            example = example[1]
        shape = PartialShape([-1] * example.ndim) if hasattr(example, "ndim") else PartialShape.dynamic()
        dtype = getattr(example, "dtype", type(example))
        dtype = pt_to_ov_type_map.get(str(dtype))
        if name:
            input_arg.append(single_input_to_input_cut_info((name, shape, dtype)))
        else:
            input_arg.append(single_input_to_input_cut_info((shape, dtype)))
    if inputs is not None and len(inputs) != 0:
        if len(inputs) == len(input_arg):
            # we can update input argument with info from examples
            new_input = []
            for i in range(len(input_arg)):
                input_desc = args.input[i]
                name = input_desc.name
                dtype = input_desc.type
                shape = input_desc.shape
                if name is None:
                    name = input_arg[i].name
                if dtype is None:
                    dtype = input_arg[i].type
                if shape is None:
                    shape = input_arg[i].shape
                new_input.append(_InputCutInfo(name, shape, dtype, input_desc.value))
            input_arg = new_input
        else:
            # we can't update args.input
            return
    args.input = input_arg


# pylint: disable=no-member
def to_torch_tensor(tensor):
    import torch  # pylint: disable=import-error
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    if isinstance(tensor, Tensor):
        return torch.tensor(tensor.data)
    if isinstance(tensor, (float, int, bool)):
        return tensor
    if isinstance(tensor, (tuple, list)):
        # TODO: Function to_torch_tensor should be renamed as it handles not only a tensor
        return tuple(to_torch_tensor(x) for x in tensor)
    if isinstance(tensor, dict) and all(isinstance(k, str) for k in tensor.keys()):
        return dict((k, to_torch_tensor(x)) for k, x in tensor.items())
    if tensor is None:
        return None
    else:
        raise Error("Unexpected type of example_input. Supported types torch.Tensor, np.array or ov.Tensor. "
                    "Got {}".format(type(tensor)))


def prepare_torch_inputs(example_inputs):
    inputs = None
    if example_inputs is not None:
        inputs = example_inputs
        if isinstance(inputs, list):
            inputs = [to_torch_tensor(x) for x in inputs]
        elif isinstance(inputs, tuple):
            inputs = [to_torch_tensor(x) for x in inputs]
            inputs = tuple(inputs)
        elif isinstance(inputs, dict):
            for name, tensor in inputs.items():
                assert isinstance(name, str), "Expected dictionary where keys are input names of string type and" \
                                              " values are tensors. Got key of type {}".format(type(name))
                inputs[name] = to_torch_tensor(tensor)
        else:
            inputs = to_torch_tensor(inputs)
    else:
        # No example_input were provided, decoder will use scripting
        return None
    return inputs
