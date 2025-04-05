# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import os
import torch
import torch.overrides

from hashlib import sha256
from torch.fx import GraphModule

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino import Core, Type, PartialShape, serialize
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_config, _is_cache_dir_in_config

from typing import Callable, Optional

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def cached_model_name(model_hash_str, device, args, cache_root, reversed = False):
    if model_hash_str is None:
        return None

    model_cache_dir = cache_root + "/model/"

    try:
        os.makedirs(model_cache_dir, exist_ok=True)
        file_name = model_cache_dir + model_hash_str + "_" + device
    except OSError as error:
        logger.warning(f"Cache directory {cache_root} cannot be created. Model caching is disabled. Error: {error }")
        return None

    inputs_str = ""
    for idx, input_data in enumerate(args):
        if reversed:
            inputs_str = "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "") + inputs_str
        else:
            inputs_str += "_" + str(input_data.type()) + str(input_data.size())[11:-1].replace(" ", "")
    inputs_str = sha256(inputs_str.encode('utf-8')).hexdigest()
    file_name += inputs_str

    return file_name

def openvino_compile_cached_model(cached_model_path, options, *example_inputs):
    core = Core()
    om = core.read_model(cached_model_path + ".xml")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(example_inputs):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    config = {}

    if _is_cache_dir_in_config(options):
        config = _get_config(options)
    else:
        config["CACHE_DIR"] = _get_cache_dir(options)

    compiled_model = core.compile_model(om, _get_device(options), config)

    return compiled_model

def openvino_compile(gm: GraphModule, *args, model_hash_str: str = None, options=None):
    core = Core()

    device = _get_device(options)
    cache_root = _get_cache_dir(options)
    file_name = cached_model_name(model_hash_str, device, args, cache_root)

    if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for idx, input_data in enumerate(args):
            if isinstance(input_data, int):
                input_types.append(torch.int64)
                input_shapes.append(torch.Size([1]))
            else:
                input_types.append(input_data.type())
                input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm)

        im = fe.load(decoder)

        om = fe.convert(im)

        if file_name is not None:
            serialize(om, file_name + ".xml", file_name + ".bin")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(args):
        if isinstance(input_data, int):
            om.inputs[idx].get_node().set_element_type(dtype_mapping[torch.int64])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(torch.Size([1]))))
        else:
            om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
            om.inputs[idx].get_node().set_partial_shape(PartialShape(list(decoder.input_shapes[idx])))

    om.validate_nodes_and_infer_types()

    config = _get_config(options)

    if model_hash_str is not None:
        if not _is_cache_dir_in_config(options):
            config["CACHE_DIR"] = cache_root

    compiled = core.compile_model(om, device, config)
    return compiled
