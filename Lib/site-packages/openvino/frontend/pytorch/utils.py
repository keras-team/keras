# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import inspect
import logging
import torch
import numpy as np

from openvino import op, Type as OVType, Shape, Tensor
from openvino import opset11 as ops

log = logging.getLogger(__name__)


def make_constant(*args, **kwargs):
    return op.Constant(*args, **kwargs)


def fetch_attr(self_module, target: str):
    """
    Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

    Args:
        target (str): The fully-qualified name of the attribute to fetch

    Return:
        Any: The value of the attribute.
    """
    target_atoms = target.split(".")
    attr_itr = self_module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def get_type_from_py_type(value):
    if isinstance(value, float):
        return OVType.f32
    if isinstance(value, bool):
        return OVType.boolean
    if isinstance(value, int):
        return OVType.i64
    return OVType.dynamic


def torch_tensor_to_ov_const(torch_t: torch.Tensor, shared_memory=True):
    is_fake_tensor = False
    try:
        from torch._prims import FakeTensor
        is_fake_tensor = isinstance(torch_t, FakeTensor)
    except:
        pass
    assert not is_fake_tensor, '`FakeTensor` is found in the graph during conversion. ' \
                               'In order to avoid `FakeTensor` in the traced model, ' \
                               'try to infer the model before exporting.'
    torch_t = torch_t.contiguous()
    if torch_t.dtype == torch.bfloat16:
        # reinterpret bfloat16 data as float16 to allow conversion to numpy
        torch_t = torch_t.view(torch.float16)
        narr = torch_t.numpy(force=True)
        tensor = Tensor(narr, torch_t.shape, OVType.bf16)
        ov_const = op.Constant(tensor, shared_memory=shared_memory)
    else:
        narr = torch_t.numpy(force=True)
        ov_const = op.Constant(narr, shared_memory=shared_memory)
    return ov_const


def ivalue_to_constant(ivalue, shared_memory=True):
    ov_type = get_type_from_py_type(ivalue)
    if ov_type.is_static():
        return op.Constant(ov_type, Shape([]), [ivalue]).outputs()

    if isinstance(ivalue, (list, tuple)):
        assert len(ivalue) > 0, "Can't deduce type for empty list"
        ov_type = get_type_from_py_type(ivalue[0])
        assert ov_type.is_static(), "Can't deduce type for list"
        return op.Constant(ov_type, Shape([len(ivalue)]), ivalue).outputs()

    if isinstance(ivalue, torch.Tensor):
        return torch_tensor_to_ov_const(ivalue, shared_memory=shared_memory).outputs()
    return None


def get_value_from_getattr(getattr_node, self_module):
    assert getattr_node.kind() == "prim::GetAttr", "Got node of kind not equal to prim::GetAttr"
    # GetAttr nodes can be nested
    stack = []
    while getattr_node.kind() == "prim::GetAttr":
        stack.append(getattr_node)
        inputs = list(getattr_node.inputs())
        if len(inputs) == 0:
            break
        getattr_node = inputs[0].node()
    module = self_module
    path_name = "self"
    while len(stack) > 0:
        node = stack.pop()
        attr_name = node.s("name")
        assert hasattr(
            module, attr_name), f'No attribute with name "{attr_name}" found in module.'
        path_name = ".".join([path_name, attr_name])
        module = getattr(module, attr_name)
    return module, path_name


def graph_has_ops(graph, op_types: list) -> bool:
    res = False
    for n in graph.nodes():
        if any(kind in n.kind() for kind in op_types):
            return True
        for b in n.blocks():
            res = graph_has_ops(b, op_types)
        if res:
            return res
    return res


pt_to_ov_type_map = {
    "float": OVType.f32,
    "int": OVType.i64,
    "bool": OVType.boolean,
    "torch.bfloat16": OVType.bf16,
    "torch.float16": OVType.f16,
    "torch.float32": OVType.f32,
    "torch.float64": OVType.f64,
    "torch.uint8": OVType.u8,
    "torch.int8": OVType.i8,
    "torch.int16": OVType.i16,
    "torch.int32": OVType.i32,
    "torch.int64": OVType.i64,
    "torch.bool": OVType.boolean,
    "torch.DoubleTensor": OVType.f64,
    "torch.FloatTensor": OVType.f32,
    "torch.HalfTensor": OVType.f16,
    "torch.BFloat16Tensor": OVType.bf16,
    "torch.IntTensor": OVType.i32,
    "torch.LongTensor": OVType.i64,
    "torch.ShortTensor": OVType.i16,
    "torch.CharTensor": OVType.i8,
    "torch.ByteTensor": OVType.u8,
    "torch.BoolTensor": OVType.boolean,
    "torch.quint8": OVType.u8,
    "torch.qint8": OVType.i8,
    "torch.qint32": OVType.i32,
}


wrapper_template = """
import torch
from typing import *

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, {input_sign}):
        return self.model({example_input})
"""


def build_wrapper(template, model):
    """
    Builds a wrapper around the given model using the provided template.
    """
    result = {}
    try:
        exec(template, result)

        wrapped_model = result["ModelWrapper"](model)
        wrapped_model.eval()
    # if wrapping failed, it is better to return original model for avoid user confusion regarding error message
    except Exception:
        log.error("Failed to build model wrapper.")
        wrapped_model = model
    return wrapped_model


def process_dict_inputs(inputs, input_params, model):
    ordered_inputs = []
    for input_name in input_params:
        if input_name in inputs:
            ordered_inputs.append(input_name)

    input_signature = list(input_params)
    if ordered_inputs == input_signature[: len(ordered_inputs)]:
        example_inputs = [inputs[input_name] for input_name in ordered_inputs]
        if all([isinstance(inp, torch.Tensor) for inp in example_inputs]):
            return {"example_inputs": [inputs[name] for name in ordered_inputs]}, ordered_inputs, model
        return {"example_inputs": example_inputs}, ordered_inputs, model

    # PyTorch has some difficulties to trace models with named unordered parameters:
    # torch < 2.0.0 supports only positional arguments for tracing
    # pytorch == 2.0.0 supports input kwargs tracing,
    # but does not support complex nested objects (e. g. tuple of tuples of tensors)
    # We will use wrapper for making them positional as workaround.

    input_sign_str = []
    input_params_str = []

    for input_name in ordered_inputs:
        if str(input_params[input_name].annotation).startswith("typing.Union"):
            filter_custom_args = []
            for arg in input_params[input_name].annotation.__args__:
                str_arg = str(arg)
                is_typing = str_arg.startswith("typing.")
                is_torch = "torch." in str_arg
                is_builten = str_arg in (str(int), str(float), str(type(None)))
                if not (is_typing or is_torch or is_builten):
                    continue
                filter_custom_args.append(arg)
            input_params[input_name].annotation.__args__ = tuple(
                filter_custom_args)
        input_sign_str.append(
            str(input_params[input_name]).replace("NoneType", "None"))
        input_params_str.append(f"{input_name}={input_name}")

    wrapper_class = wrapper_template.format(input_sign=", ".join(
        input_sign_str), example_input=", ".join(input_params_str))

    wrapped_model = build_wrapper(wrapper_class, model)

    return {"example_inputs": [inputs[name] for name in ordered_inputs]}, ordered_inputs, wrapped_model


def prepare_example_inputs_and_model(inputs, input_params, model):
    input_is_list = False
    input_signature = list(input_params)
    if isinstance(inputs, dict):
        examples, ordered, wrapped = process_dict_inputs(
            inputs, input_params, model)
        return examples, ordered, wrapped, input_is_list
    if isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], torch.Tensor):
        if "typing.List" in str(input_params[input_signature[0]].annotation):
            inputs = inputs[0].unsqueeze(0)
            input_is_list = True

    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    input_signature = input_signature[: len(inputs)]
    return {"example_inputs": inputs}, input_signature, model, input_is_list


def convert_quantized_tensor(qtensor: torch.Tensor, shared_memory: bool):
    # represents torch quantized tensor as
    # Constant(u8) -> Convert(f32) -> Subtract(zero_point) -> Multiply(scale)
    qscheme = qtensor.qscheme()
    if qscheme == torch.per_channel_affine:
        int8_tensor = qtensor.int_repr()
        scale = qtensor.q_per_channel_scales().numpy().astype(np.float32)
        zero_point = qtensor.q_per_channel_zero_points().numpy().astype(np.float32)
        axis = np.int32(qtensor.q_per_channel_axis())

        new_shape = np.ones(len(int8_tensor.shape), dtype=np.int32)
        new_shape[axis] = -1
        zero_point_bc = np.reshape(zero_point, new_shape)
        scale_bc = np.reshape(scale, new_shape)

        int8_const = torch_tensor_to_ov_const(
            int8_tensor, shared_memory=shared_memory)
        convert = ops.convert(int8_const, np.float32)
        sub = ops.subtract(convert, zero_point_bc)
        return ops.multiply(sub, scale_bc).outputs()
    elif qscheme == torch.per_tensor_affine:
        int8_tensor = qtensor.int_repr()
        scale = np.float32(qtensor.q_scale())
        zero_point = np.float32(qtensor.q_zero_point())

        int8_const = torch_tensor_to_ov_const(
            int8_tensor, shared_memory=shared_memory)
        convert = ops.convert(int8_const, np.float32)
        sub = ops.subtract(convert, zero_point)
        return ops.multiply(sub, scale).outputs()
    assert False, "Unsupported qscheme"


def process_individual_input(x, x_name):
    """
    Processes an individual input and generates a signature,
    parameter string, example entry, and a wrap flag.
    """
    sign = None
    param = None
    example_entry = None
    to_wrap = False
    if isinstance(x, tuple):
        internal_input = []
        new_tuple = []
        index = 0
        for v in x:
            if v is None:
                to_wrap = True
                internal_input.append("None")
            else:
                internal_input.append(f"{x_name}[{index}]")
                new_tuple.append(v)
                index += 1
        param = f"({', '.join(internal_input)},)"
        if len(new_tuple) > 0:
            example_entry = tuple(new_tuple)
            sign = x_name
    elif x is None:
        to_wrap = True
        param = "None"
    else:
        sign = x_name
        param = x_name
        example_entry = x
    return sign, param, example_entry, to_wrap


def patch_none_example(model: torch.nn.Module, example):
    """
    Patches a PyTorch model to handle None values in the input example.
    """
    callable_func = getattr(model, "forward", model.__call__)
    input_params = inspect.signature(callable_func).parameters
    input_signature = list(input_params)
    input_sign_str = []
    input_params_str = []
    to_wrap = False
    if isinstance(example, tuple) and len(input_signature) >= len(example):
        new_example = []
        for i, x in enumerate(example):
            x_name = input_signature[i]
            sign, param, example_entry, _to_wrap = process_individual_input(x, x_name)
            to_wrap = to_wrap or _to_wrap
            if sign is not None:
                input_sign_str.append(str(input_params[sign]))
            input_params_str.append(param)
            if example_entry is not None:
                new_example.append(example_entry)
        if to_wrap:
            wrapper_class = wrapper_template.format(input_sign=", ".join(input_sign_str),
                                                    example_input=", ".join(input_params_str))
            wrapped_model = build_wrapper(wrapper_class, model)
            log.warning("Model has None in the example input. The input "
                        "with None will be removed from the resulting model.")
            return wrapped_model, tuple(new_example)
    elif isinstance(example, dict) and len(input_signature) >= len(example):
        new_example = {}
        input_signature = [s for s in input_signature if s in example]
        for x_name in input_signature:
            x = example[x_name]
            sign, param, example_entry, _to_wrap = process_individual_input(x, x_name)
            to_wrap = to_wrap or _to_wrap
            if sign is not None:
                input_sign_str.append(str(input_params[sign]))
            input_params_str.append(f"{x_name}={param}")
            if example_entry is not None:
                new_example[x_name] = example_entry
        if to_wrap:
            wrapper_class = wrapper_template.format(input_sign=", ".join(input_sign_str),
                                                    example_input=", ".join(input_params_str))
            wrapped_model = build_wrapper(wrapper_class, model)
            log.warning("Model has None in the example input. The input "
                        "with None will be removed from the resulting model.")
            return wrapped_model, new_example
    return model, example
