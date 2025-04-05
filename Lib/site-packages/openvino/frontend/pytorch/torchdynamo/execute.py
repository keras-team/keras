# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from warnings import warn

import torch
import torch.overrides

from torch.fx import GraphModule
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.compile import openvino_compile
from openvino import Core, Type, PartialShape
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_aot_autograd

from typing import Callable, Optional, Any

from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


DEFAULT_OPENVINO_PYTHON_CONFIG = MappingProxyType(
    {
        "use_python_fusion_cache": True,
        "allow_single_op_fusion": True,
    },
)

compiled_cache = {}
req_cache = {}
max_openvino_partitions = 0
partitioned_modules = {}


def execute(
    gm: GraphModule,
    *args,
    executor: str = "openvino",
    executor_parameters: Optional[dict] = None,
    options: Optional[Any] = None,
):
    if executor == "openvino":
        return openvino_execute_partitioned(gm, *args, executor_parameters=executor_parameters, options=options)
    elif executor == "strictly_openvino":
        return openvino_execute(gm, *args, executor_parameters=executor_parameters)

    msg = "Received unexpected value for 'executor': {0}. Allowed values are: openvino, strictly_openvino.".format(executor)
    raise ValueError(msg)


import numpy as np


def execute_cached(compiled_model, *args):
    ov_inputs = [a.detach().cpu().numpy() for a in args]
    ov_inputs.reverse()
    res = compiled_model(ov_inputs)
    result = [torch.from_numpy(res[out]) for out in compiled_model.outputs]
    return result


def openvino_execute(gm: GraphModule, *args, executor_parameters=None, partition_id, options):

    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    use_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    global compiled_cache

    model_hash_str = executor_parameters.get("model_hash_str", None)
    if model_hash_str is not None:
        fully_supported = False
        if len(model_hash_str) > 3 and model_hash_str[-3:] == "_fs":
            fully_supported = True
        if not fully_supported:
            model_hash_str = model_hash_str + "_p" + str(partition_id)

    if use_cache and (partition_id in compiled_cache):
        compiled = compiled_cache[partition_id]
        req = req_cache[partition_id]
    else:
        compiled = openvino_compile(gm, *args, model_hash_str=model_hash_str, options=options)
        compiled_cache[partition_id] = compiled
        req = compiled.create_infer_request()
        req_cache[partition_id] = req

    flat_args, _ = tree_flatten(args)
    ov_inputs = []
    for arg in flat_args:
        ov_inputs.append((arg if isinstance(arg, int) else arg.detach().cpu().numpy()))

    res = req.infer(ov_inputs, share_inputs=True, share_outputs=True)

    results1 = [torch.from_numpy(res[out]) for out in compiled.outputs]
    if len(results1) == 1:
        return results1[0]
    return results1


class OpenVINOGraphModule(torch.nn.Module):
    def __init__(self, gm, partition_id, use_python_fusion_cache, model_hash_str: str = None, options=None):
        super().__init__()
        self.gm = gm
        self.partition_id = partition_id
        self.executor_parameters = {"use_python_fusion_cache": use_python_fusion_cache,
                                    "model_hash_str": model_hash_str}
        self.perm_fallback = False
        self.options = options

    def __call__(self, *args):
        if self.perm_fallback:
            return self.gm(*args)

        try:
            result = openvino_execute(self.gm, *args, executor_parameters=self.executor_parameters, partition_id=self.partition_id, options=self.options)
        except Exception:
            logger.debug("OpenVINO execution failed. Falling back to native PyTorch execution.")
            self.perm_fallback = True
            return self.gm(*args)

        return result


def partition_graph(gm: GraphModule, use_python_fusion_cache: bool, model_hash_str: str = None, options=None):
    global max_openvino_partitions
    partition_id = max_openvino_partitions
    for node in gm.graph.nodes:
        # TODO: use a better way to identify fused submodule
        if node.op == "call_module" and "fused_" in node.name:
            openvino_submodule = getattr(gm, node.name)
            gm.delete_submodule(node.target)
            gm.add_submodule(
                node.target,
                OpenVINOGraphModule(openvino_submodule, partition_id, use_python_fusion_cache,
                                    model_hash_str=model_hash_str, options=options),
            )
            partition_id = partition_id + 1

    max_openvino_partitions = partition_id

    return gm


def openvino_execute_partitioned(gm: GraphModule, *args, executor_parameters=None, options=None):
    executor_parameters = executor_parameters or DEFAULT_OPENVINO_PYTHON_CONFIG

    global partitioned_modules

    use_python_fusion_cache = executor_parameters.get(
        "use_python_fusion_cache",
        DEFAULT_OPENVINO_PYTHON_CONFIG["use_python_fusion_cache"],
    )
    model_hash_str = executor_parameters.get("model_hash_str", None)

    signature = str(id(gm))
    if (not _get_aot_autograd(options)):
        for idx, input_data in enumerate(args):
            if isinstance(input_data, torch.Tensor):
                signature = signature + "_" + str(idx) + ":" + str(input_data.type())[6:] + ":" + str(input_data.size())[11:-1].replace(" ", "")
            else:
                signature = signature + "_" + str(idx) + ":" + type(input_data).__name__ + ":val(" + str(input_data) + ")"

    if signature not in partitioned_modules:
        partitioned_modules[signature] = partition_graph(gm, use_python_fusion_cache=use_python_fusion_cache,
                                                         model_hash_str=model_hash_str, options=options)
    return partitioned_modules[signature](*args)


def clear_caches():
    global partitioned_modules
    global compiled_cache

    compiled_cache.clear()
    partitioned_modules.clear()
