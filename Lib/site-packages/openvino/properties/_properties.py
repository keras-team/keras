# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from types import BuiltinFunctionType, ModuleType
from typing import Callable, Any, Union


class Property(str):
    """This class allows to make a string object callable. Call returns underlying string's data."""
    def __new__(cls, prop: Callable[..., Any]):  # type: ignore
        instance = super().__new__(cls, prop())
        instance.prop = prop
        return instance

    def __call__(self, *args: Any) -> Callable[..., Any]:
        if args is not None:
            from openvino import Model
            if args and isinstance(args[0], Model):
                return self.prop(args[0]._Model__model)
            return self.prop(*args)
        return self.prop()


def __append_property_to_module(func: Callable[..., Any], target_module_name: str) -> None:
    """Modifies the target module's __getattr__ method to expose a python property wrapper by the function's name.

    :param func: the function which will be transformed to behave as python property.
    :param target_module_name: the name of the module to which properties are added.
    """
    module = sys.modules[target_module_name]

    def base_getattr(name: str) -> None:
        raise AttributeError(
            f"Module '{module.__name__}' doesn't have the attribute with name '{name}'.")

    getattr_old = getattr(module, "__getattr__", base_getattr)

    def getattr_new(name: str) -> Union[Callable[..., Any], Any]:
        if func.__name__ == name:
            return Property(func)
        else:
            return getattr_old(name)

    module.__getattr__ = getattr_new  # type: ignore


def __make_properties(source_module_of_properties: ModuleType, target_module_name: str) -> None:
    """Makes python properties in target module from functions found in the source module.

    :param source_module_of_properties: the source module from which functions should be taken.
    :param target_module_name: the name of the module to which properties are added.
    """
    for attr in dir(source_module_of_properties):
        func = getattr(source_module_of_properties, attr)
        if isinstance(func, BuiltinFunctionType):
            __append_property_to_module(func, target_module_name)
