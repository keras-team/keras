# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from inspect import signature
from typing import Any, Callable, Dict, Optional, Union, get_origin, get_args

from openvino import Node, Output
from openvino.utils.types import NodeInput, as_node, as_nodes


def _get_name(**kwargs: Any) -> Node:
    if "name" in kwargs:
        return kwargs["name"]
    return None


def _set_node_friendly_name(node: Node, *, name: Optional[str] = None) -> Node:
    if name is not None:
        node.friendly_name = name
    return node


def nameable_op(node_factory_function: Callable) -> Callable:
    """Set the name to the openvino operator returned by the wrapped function."""

    @wraps(node_factory_function)
    def wrapper(*args: Any, **kwargs: Any) -> Node:
        node = node_factory_function(*args, **kwargs)
        node = _set_node_friendly_name(node, name=_get_name(**kwargs))
        return node

    return wrapper


def unary_op(node_factory_function: Callable) -> Callable:
    """Convert the first input value to a Constant Node if a numeric value is detected."""

    @wraps(node_factory_function)
    def wrapper(input_value: NodeInput, *args: Any, **kwargs: Any) -> Node:
        input_node = as_node(input_value, name=_get_name(**kwargs))
        node = node_factory_function(input_node, *args, **kwargs)
        node = _set_node_friendly_name(node, name=_get_name(**kwargs))
        return node

    return wrapper


def binary_op(node_factory_function: Callable) -> Callable:
    """Convert the first two input values to Constant Nodes if numeric values are detected."""

    @wraps(node_factory_function)
    def wrapper(left: NodeInput, right: NodeInput, *args: Any, **kwargs: Any) -> Node:
        left, right = as_nodes(left, right, name=_get_name(**kwargs))
        node = node_factory_function(left, right, *args, **kwargs)
        node = _set_node_friendly_name(node, name=_get_name(**kwargs))
        return node

    return wrapper


def custom_preprocess_function(custom_function: Callable) -> Callable:
    """Convert Node returned from custom_function to Output."""

    @wraps(custom_function)
    def wrapper(node: Node) -> Output:
        return Output._from_node(custom_function(node))

    return wrapper


class MultiMethod(object):
    def __init__(self, name: str):
        self.name = name
        self.typemap: Dict[tuple, Callable] = {}

    # Checks if actual_type is a subclass of any type in the union
    def matches_union(self, union_type, actual_type) -> bool:  # type: ignore
        for type_arg in get_args(union_type):
            if isinstance(type_arg, type) and issubclass(actual_type, type_arg):
                return True
            elif get_origin(type_arg) == list:
                if issubclass(actual_type, list):
                    return True
        return False

    def matches_optional(self, optional_type, actual_type) -> bool:  # type: ignore
        return actual_type is None or self.matches_union(optional_type, actual_type)

    # Checks whether there is overloading which matches invoked argument types
    def check_invoked_types_in_overloaded_funcs(self, tuple_to_check: tuple, key_structure: tuple) -> bool:
        for actual_type, expected_type in zip(tuple_to_check, key_structure):
            origin = get_origin(expected_type)
            if origin is Union:
                if not self.matches_union(expected_type, actual_type):
                    return False
            elif origin is Optional:
                if not self.matches_optional(expected_type, actual_type):
                    return False
            elif not issubclass(actual_type, expected_type):
                return False
        return True

    def __call__(self, *args, **kwargs) -> Any:  # type: ignore
        arg_types = tuple(arg.__class__ for arg in args)
        kwarg_types = {key: type(value) for key, value in kwargs.items()}

        key_matched = None
        if len(kwarg_types) == 0 and len(arg_types) != 0:
            for key in self.typemap.keys():
                # compare types of called function with overloads
                if self.check_invoked_types_in_overloaded_funcs(arg_types, key):
                    key_matched = key
                    break
        elif len(arg_types) == 0 and len(kwarg_types) != 0:
            for key, func in self.typemap.items():
                func_signature = {arg_name: types.annotation for arg_name, types in signature(func).parameters.items()}
                # if kwargs of called function are subset of overloaded function, we use this overload
                if kwarg_types.keys() <= func_signature.keys():
                    key_matched = key
                    break
        elif len(arg_types) != 0 and len(kwarg_types) != 0:
            for key, func in self.typemap.items():
                func_signature = {arg_name: types.annotation for arg_name, types in signature(func).parameters.items()}
                # compare types of called function with overloads
                if self.check_invoked_types_in_overloaded_funcs(arg_types, tuple(func_signature.values())):
                    # if kwargs of called function are subset of overloaded function, we use this overload
                    if kwarg_types.keys() <= func_signature.keys():
                        key_matched = key
                        break

        if key_matched is None:
            raise TypeError(f"The necessary overload for {self.name} was not found")

        function = self.typemap.get(key_matched)
        return function(*args, **kwargs)  # type: ignore

    def register(self, types: tuple, function: Callable) -> None:
        if types in self.typemap:
            raise TypeError("duplicate registration")
        self.typemap[types] = function


registry: Dict[str, MultiMethod] = {}


def overloading(*types: tuple) -> Callable:
    def register(function: Callable) -> MultiMethod:
        name = function.__name__
        mm = registry.get(name)
        if mm is None:
            mm = registry[name] = MultiMethod(name)
        mm.register(types, function)
        return mm
    return register
