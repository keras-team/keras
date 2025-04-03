# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import os
from functools import partial
from hashlib import sha256

import torch
from torch._dynamo.backends.common import fake_tensor_unsupported, aot_autograd
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx
from torch._inductor.freezing import replace_params_with_constants
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import decomposition_table, get_decompositions

from openvino.frontend import FrontEndManager
from openvino import Core, Type, PartialShape
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.torchdynamo import decompositions
from openvino.frontend.pytorch.torchdynamo.decompositions import get_aot_decomposition_list, get_inf_decomposition_list
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute, execute_cached
from openvino.frontend.pytorch.torchdynamo.compile import cached_model_name, openvino_compile_cached_model
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_model_caching, _get_decompositions, _get_aot_autograd

from openvino import Core, Type, PartialShape

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

"""
    This is a preview feature in OpenVINO. This feature
    enables users to compile PyTorch models using torch.compile
    with OpenVINO as a target backend in PyTorch applications

    Sample usage:
    This sample code loads resnet50 torchvision model and compiles it using torch dynamo.
    We can then use this model for inference. We only need to add two lines of code to
    the Pytorch applications which are marked in the code below

    1) import openvino.torch
    model = torchvision.models.resnet50()
    2) model = torch.compile(model, backend="openvino")
"""

openvino_options = {}

# Disable regional compilation which was enabled by default from Torch 2.5.0
if hasattr(torch._dynamo.config, "inline_inbuilt_nn_modules"):
    torch._dynamo.config.inline_inbuilt_nn_modules=False

@fake_tensor_unsupported
def openvino(subgraph, example_inputs, options=None):
    if _get_aot_autograd(options):
        global openvino_options
        openvino_options = options
        decompositions = _get_decompositions(options) + get_inf_decomposition_list() + get_aot_decomposition_list()
        return aot_autograd(fw_compiler=fx_openvino, bw_compiler=fx_openvino, decompositions=get_decompositions(decompositions))(subgraph, example_inputs)
    return fx_openvino(subgraph, example_inputs, options)

if "openvino" not in torch.compiler.list_backends():
    register_backend(compiler_fn=openvino, name="openvino")

def fx_openvino(subgraph, example_inputs, options=None):
    try:
        if len(openvino_options) != 0:
            options = openvino_options
        executor_parameters = None
        inputs_reversed = False
        openvino_model_caching = _get_model_caching(options)
        if openvino_model_caching is not None and openvino_model_caching:
            # Create a hash to be used for caching
            model_hash_str = sha256(subgraph.code.encode("utf-8")).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", _get_device(options), example_inputs, _get_cache_dir(options))
            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, options, *example_inputs)

                def _call(*args):
                    res = execute_cached(compiled_model, *args)
                    return res

                return _call
        if inputs_reversed:
            example_inputs.reverse()

        preserved_arg_indices = []
        if _get_aot_autograd(options):
            if tracing_context := torch._guards.TracingContext.try_get():
                fw_metadata = tracing_context.fw_metadata
                params_flat = tracing_context.params_flat
                assert fw_metadata is not None and params_flat is not None
            preserved_arg_indices = replace_params_with_constants(subgraph, params_flat, fw_metadata)
            example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
            model = subgraph
        else:
            from torch._subclasses.fake_tensor import FakeTensorMode

            decompositions = _get_decompositions(options) + get_inf_decomposition_list()
            with FakeTensorMode(allow_non_fake_inputs=True):
                model = make_fx(subgraph, decomposition_table=get_decompositions(decompositions))(*example_inputs)

            with torch.no_grad():
                model.eval()
        partitioner = Partitioner(options)
        compiled_model = partitioner.make_partitions(model, options)

        if executor_parameters is not None and "model_hash_str" in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            if _get_aot_autograd(options):
                args_list = args[0]
                args_new = [args_list[i] for i in preserved_arg_indices]
                args = args_new
            res = execute(compiled_model, *args, executor="openvino", executor_parameters=executor_parameters, options=options)
            return res

        if _get_aot_autograd(options):
            _call._boxed_call = True  # type: ignore[attr-defined]
        return _call
    except Exception as e:
        logger.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)


def reset():
    clear_caches()
