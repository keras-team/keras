# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch
from typing import Any, Dict, Union, Optional

import numpy as np

from openvino._pyopenvino import ConstOutput, Tensor, Type, RemoteTensor
from openvino.utils.data_helpers.wrappers import _InferRequestWrapper, OVDict

ContainerTypes = Union[dict, list, tuple, OVDict]
ScalarTypes = Union[np.number, int, float]
ValidKeys = Union[str, int, ConstOutput]


def is_list_simple_type(input_list: list) -> bool:
    for sublist in input_list:
        if isinstance(sublist, list):
            for element in sublist:
                if not isinstance(element, (str, float, int, bytes)):
                    return False
        else:
            if not isinstance(sublist, (str, float, int, bytes)):
                return False
    return True


def get_request_tensor(
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    if key is None:
        return request.get_input_tensor()
    elif isinstance(key, int):
        return request.get_input_tensor(key)
    elif isinstance(key, (str, ConstOutput)):
        return request.get_tensor(key)
    else:
        raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")


@singledispatch
def value_to_tensor(
    value: Union[Tensor, np.ndarray, ScalarTypes, str],
    request: Optional[_InferRequestWrapper] = None,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> None:
    raise TypeError(f"Incompatible inputs of type: {type(value)}")


@value_to_tensor.register(Tensor)
def _(
    value: Tensor,
    request: Optional[_InferRequestWrapper] = None,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    return value


@value_to_tensor.register(RemoteTensor)
def _(
    value: RemoteTensor,
    request: Optional[_InferRequestWrapper] = None,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> RemoteTensor:
    return value


@value_to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    request: _InferRequestWrapper,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    tensor = get_request_tensor(request, key)
    tensor_type = tensor.get_element_type()
    tensor_dtype = tensor_type.to_dtype()
    # String edge-case, always copy.
    # Scalars are also handled by C++.
    if tensor_type == Type.string:
        return Tensor(value, shared_memory=False)
    # Scalars edge-case:
    if value.ndim == 0:
        tensor_shape = tuple(tensor.shape)
        if tensor_dtype == value.dtype and tensor_shape == value.shape:
            return Tensor(value, shared_memory=is_shared)
        elif tensor.size == 0:
            # the first infer request for dynamic input cannot reshape to 0 shape
            return Tensor(value.astype(tensor_dtype).reshape((1)), shared_memory=False)
        else:
            return Tensor(value.astype(tensor_dtype).reshape(tensor_shape), shared_memory=False)
    # WA for FP16-->BF16 edge-case, always copy.
    if tensor_type == Type.bf16:
        tensor = Tensor(tensor_type, value.shape)
        tensor.data[:] = value.view(tensor_dtype)
        return tensor
    # WA for "not writeable" edge-case, always copy.
    if value.flags["WRITEABLE"] is False:
        tensor = Tensor(tensor_type, value.shape)
        tensor.data[:] = value.astype(tensor_dtype) if tensor_dtype != value.dtype else value
        return tensor
    # If types are mismatched, convert and always copy.
    if tensor_dtype != value.dtype:
        return Tensor(value.astype(tensor_dtype), shared_memory=False)
    # Otherwise, use mode defined in the call.
    return Tensor(value, shared_memory=is_shared)


@value_to_tensor.register(list)
def _(
    value: list,
    request: _InferRequestWrapper,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    return Tensor(value)


@value_to_tensor.register(np.number)
@value_to_tensor.register(int)
@value_to_tensor.register(float)
@value_to_tensor.register(str)
@value_to_tensor.register(bytes)
def _(
    value: Union[ScalarTypes, str, bytes],
    request: _InferRequestWrapper,
    is_shared: bool = False,
    key: Optional[ValidKeys] = None,
) -> Tensor:
    # np.number/int/float/str/bytes edge-case, copy will occur in both scenarios.
    tensor_type = get_request_tensor(request, key).get_element_type()
    tensor_dtype = tensor_type.to_dtype()
    tmp = np.array(value)
    # String edge-case -- it converts the data inside of Tensor class.
    # If types are mismatched, convert.
    if tensor_type != Type.string and tensor_dtype != tmp.dtype:
        return Tensor(tmp.astype(tensor_dtype), shared_memory=False)
    return Tensor(tmp, shared_memory=False)


def to_c_style(value: Any, is_shared: bool = False) -> Any:
    if not isinstance(value, np.ndarray):
        if hasattr(value, "__array__"):
            if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
                # https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
                return to_c_style(np.asarray(value), is_shared) if is_shared else np.asarray(value, copy=True)  # type: ignore
            else:
                return to_c_style(np.array(value, copy=False), is_shared) if is_shared else np.array(value, copy=True)
        return value
    return value if value.flags["C_CONTIGUOUS"] else np.ascontiguousarray(value)


###
# Start of array normalization.
###
@singledispatch
def normalize_arrays(
    inputs: Any,
    is_shared: bool = False,
) -> Any:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            # https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
            return to_c_style(np.asarray(inputs), is_shared) if is_shared else np.asarray(inputs, copy=True)  # type: ignore
        else:
            return to_c_style(np.array(inputs, copy=False), is_shared) if is_shared else np.array(inputs, copy=True)
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@normalize_arrays.register(dict)
def _(
    inputs: dict,
    is_shared: bool = False,
) -> dict:
    return {k: to_c_style(v, is_shared) if is_shared else v for k, v in inputs.items()}


@normalize_arrays.register(OVDict)
def _(
    inputs: OVDict,
    is_shared: bool = False,
) -> dict:
    return {i: to_c_style(v, is_shared) if is_shared else v for i, (_, v) in enumerate(inputs.items())}


@normalize_arrays.register(list)
@normalize_arrays.register(tuple)
def _(
    inputs: Union[list, tuple],
    is_shared: bool = False,
) -> dict:
    return {i: to_c_style(v, is_shared) if is_shared else v for i, v in enumerate(inputs)}


@normalize_arrays.register(np.ndarray)
def _(
    inputs: dict,
    is_shared: bool = False,
) -> Any:
    return to_c_style(inputs, is_shared) if is_shared else inputs
###
# End of array normalization.
###


###
# Start of "shared" dispatcher.
# (1) Each method should keep Tensors "as-is", regardless to them being shared or not.
# (2) ...
###
# Step to keep alive input values that are not C-style by default
@singledispatch
def create_shared(
    inputs: Any,
    request: _InferRequestWrapper,
) -> None:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        request._inputs_data = normalize_arrays(inputs, is_shared=True)
        return value_to_tensor(request._inputs_data, request=request, is_shared=True)
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@create_shared.register(dict)
@create_shared.register(tuple)
@create_shared.register(OVDict)
def _(
    inputs: Union[dict, tuple, OVDict],
    request: _InferRequestWrapper,
) -> dict:
    request._inputs_data = normalize_arrays(inputs, is_shared=True)
    return {k: value_to_tensor(v, request=request, is_shared=True, key=k) for k, v in request._inputs_data.items()}


# Special override to perform list-related dispatch
@create_shared.register(list)
def _(
    inputs: list,
    request: _InferRequestWrapper,
) -> dict:
    # If list is passed to single input model and consists only of simple types
    # i.e. str/bytes/float/int, wrap around it and pass into the dispatcher.
    request._inputs_data = normalize_arrays([inputs] if request._is_single_input() and is_list_simple_type(inputs) else inputs, is_shared=True)
    return {k: value_to_tensor(v, request=request, is_shared=True, key=k) for k, v in request._inputs_data.items()}


@create_shared.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: _InferRequestWrapper,
) -> Tensor:
    request._inputs_data = normalize_arrays(inputs, is_shared=True)
    return value_to_tensor(request._inputs_data, request=request, is_shared=True)


@create_shared.register(Tensor)
@create_shared.register(np.number)
@create_shared.register(int)
@create_shared.register(float)
@create_shared.register(str)
@create_shared.register(bytes)
def _(
    inputs: Union[Tensor, ScalarTypes, str, bytes],
    request: _InferRequestWrapper,
) -> Tensor:
    return value_to_tensor(inputs, request=request, is_shared=True)
###
# End of "shared" dispatcher methods.
###


###
# Start of "copied" dispatcher.
###
def set_request_tensor(
    request: _InferRequestWrapper,
    tensor: Tensor,
    key: Optional[ValidKeys] = None,
) -> None:
    if key is None:
        request.set_input_tensor(tensor)
    elif isinstance(key, int):
        request.set_input_tensor(key, tensor)
    elif isinstance(key, (str, ConstOutput)):
        request.set_tensor(key, tensor)
    else:
        raise TypeError(f"Unsupported key type: {type(key)} for Tensor under key: {key}")


@singledispatch
def update_tensor(
    inputs: Any,
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> None:
    if hasattr(inputs, "__array__"):
        update_tensor(normalize_arrays(inputs, is_shared=False), request, key)
        return None
    raise TypeError(f"Incompatible inputs of type: {type(inputs)} under {key} key!")


@update_tensor.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> None:
    if inputs.ndim != 0:
        tensor = get_request_tensor(request, key)
        # Update shape if there is a mismatch
        if tuple(tensor.shape) != inputs.shape:
            tensor.shape = inputs.shape
        # When copying, type should be up/down-casted automatically.
        if tensor.element_type == Type.string:
            tensor.bytes_data = inputs
        else:
            tensor.data[:] = inputs[:]
    else:
        # If shape is "empty", assume this is a scalar value
        set_request_tensor(
            request,
            value_to_tensor(inputs, request=request, is_shared=False, key=key),
            key,
        )


@update_tensor.register(np.number)  # type: ignore
@update_tensor.register(float)
@update_tensor.register(int)
@update_tensor.register(str)
def _(
    inputs: Union[ScalarTypes, str],
    request: _InferRequestWrapper,
    key: Optional[ValidKeys] = None,
) -> None:
    set_request_tensor(
        request,
        value_to_tensor(inputs, request=request, is_shared=False, key=key),
        key,
    )


def update_inputs(inputs: dict, request: _InferRequestWrapper) -> dict:
    """Helper function to prepare inputs for inference.

    It creates copy of Tensors or copy data to already allocated Tensors on device
    if the item is of type `np.ndarray`, `np.number`, `int`, `float` or has numpy __array__ attribute.
    If value is of type `list`, create a Tensor based on it, copy will occur in the Tensor constructor.
    """
    # Create new temporary dictionary.
    # new_inputs will be used to transfer data to inference calls,
    # ensuring that original inputs are not overwritten with Tensors.
    new_inputs: Dict[ValidKeys, Tensor] = {}
    for key, value in inputs.items():
        if not isinstance(key, (str, int, ConstOutput)):
            raise TypeError(f"Incompatible key type for input: {key}")
        # Copy numpy arrays to already allocated Tensors.
        # If value object has __array__ attribute, load it to Tensor using np.array
        if isinstance(value, (np.ndarray, np.number, int, float, str)) or hasattr(value, "__array__"):
            update_tensor(value, request, key)
        elif isinstance(value, list):
            new_inputs[key] = Tensor(value)
        # If value is of Tensor type, put it into temporary dictionary.
        elif isinstance(value, Tensor):
            new_inputs[key] = value
        # Throw error otherwise.
        else:
            raise TypeError(f"Incompatible inputs of type: {type(value)} under {key} key!")
    return new_inputs


@singledispatch
def create_copied(
    inputs: Union[ContainerTypes, np.ndarray, ScalarTypes, str, bytes],
    request: _InferRequestWrapper,
) -> Union[dict, None]:
    # Check the special case of the array-interface
    if hasattr(inputs, "__array__"):
        update_tensor(normalize_arrays(inputs, is_shared=False), request, key=None)
        return {}
    # Error should be raised if type does not match any dispatchers
    raise TypeError(f"Incompatible inputs of type: {type(inputs)}")


@create_copied.register(dict)
@create_copied.register(tuple)
@create_copied.register(OVDict)
def _(
    inputs: Union[dict, tuple, OVDict],
    request: _InferRequestWrapper,
) -> dict:
    return update_inputs(normalize_arrays(inputs, is_shared=False), request)


# Special override to perform list-related dispatch
@create_copied.register(list)
def _(
    inputs: list,
    request: _InferRequestWrapper,
) -> dict:
    # If list is passed to single input model and consists only of simple types
    # i.e. str/bytes/float/int, wrap around it and pass into the dispatcher.
    return update_inputs(normalize_arrays([inputs] if request._is_single_input() and is_list_simple_type(inputs) else inputs, is_shared=False), request)


@create_copied.register(np.ndarray)
def _(
    inputs: np.ndarray,
    request: _InferRequestWrapper,
) -> dict:
    update_tensor(normalize_arrays(inputs, is_shared=False), request, key=None)
    return {}


@create_copied.register(Tensor)
@create_copied.register(np.number)
@create_copied.register(int)
@create_copied.register(float)
@create_copied.register(str)
@create_copied.register(bytes)
def _(
    inputs: Union[Tensor, ScalarTypes, str, bytes],
    request: _InferRequestWrapper,
) -> Tensor:
    return value_to_tensor(inputs, request=request, is_shared=False)
###
# End of "copied" dispatcher methods.
###


def _data_dispatch(
    request: _InferRequestWrapper,
    inputs: Union[ContainerTypes, Tensor, np.ndarray, ScalarTypes, str] = None,
    is_shared: bool = False,
) -> Union[dict, Tensor]:
    if inputs is None:
        return {}
    return create_shared(inputs, request) if is_shared else create_copied(inputs, request)
