# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from functools import wraps
from typing import Callable, Any
from pathlib import Path
import importlib.util
from types import ModuleType


def _add_openvino_libs_to_search_path() -> None:
    """Add OpenVINO libraries to the DLL search path on Windows."""
    if sys.platform == "win32":
        # Installer, yum, pip installs openvino dlls to the different directories
        # and those paths need to be visible to the openvino modules
        #
        # If you're using a custom installation of openvino,
        # add the location of openvino dlls to your system PATH.
        openvino_libs = []
        if os.path.isdir(os.path.join(os.path.dirname(__file__), "libs")):
            # looking for the libs in the pip installation path.
            openvino_libs.append(os.path.join(os.path.dirname(__file__), "libs"))
        elif os.path.isdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "Library", "bin")):
            # looking for the libs in the conda installation path
            openvino_libs.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "Library", "bin"))
        else:
            # setupvars.bat script set all libs paths to OPENVINO_LIB_PATHS environment variable.
            openvino_libs_installer = os.getenv("OPENVINO_LIB_PATHS")
            if openvino_libs_installer:
                openvino_libs.extend(openvino_libs_installer.split(";"))
            else:
                sys.exit("Error: Please set the OPENVINO_LIB_PATHS environment variable. "
                         "If you use an install package, please, run setupvars.bat")
        for lib in openvino_libs:
            lib_path = os.path.join(os.path.dirname(__file__), lib)
            if os.path.isdir(lib_path):
                # On Windows, with Python >= 3.8, DLLs are no longer imported from the PATH.
                os.add_dll_directory(os.path.abspath(lib_path))


def get_cmake_path() -> str:
    """Searches for the directory containing CMake files within the package install directory.

    :return: The path to the directory containing CMake files, if found. Otherwise, returns empty string.
    :rtype: str
    """
    package_path = Path(__file__).parent
    cmake_file = "OpenVINOConfig.cmake"

    for dirpath, _, filenames in os.walk(package_path):
        if cmake_file in filenames:
            return dirpath

    return ""


def deprecated(name: Any = None, version: str = "", message: str = "", stacklevel: int = 2) -> Callable[..., Any]:
    """Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}" and runs the function.

    :param version: The version in which the code will be removed.
    :param message: A message explaining why the function is deprecated and/or what to use instead.
    """

    def decorator(wrapped: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(wrapped)
        def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:
            # it must be imported here; otherwise, there are errors with no loaded DLL for Windows
            from openvino._pyopenvino.util import deprecation_warning

            deprecation_warning(wrapped.__name__ if name is None else name, version, message, stacklevel)
            return wrapped(*args, **kwargs)

        return wrapper

    return decorator


# WA method since Python 3.11 does not support @classmethod and @property chain,
# currently only read-only properties are supported.
class _ClassPropertyDescriptor(object):
    def __init__(self, fget: Callable):
        self.fget = fget

    def __get__(self, obj: Any, cls: Any = None) -> Any:
        if cls is None:
            cls = type(obj)
        return self.fget.__get__(obj, cls)()


def classproperty(func: Any) -> _ClassPropertyDescriptor:
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


def deprecatedclassproperty(name: Any = None, version: str = "", message: str = "", stacklevel: int = 2) -> Callable[[Any], _ClassPropertyDescriptor]:
    def decorator(wrapped: Any) -> _ClassPropertyDescriptor:
        func = classproperty(wrapped)

        # Override specific instance
        def _patch(instance: _ClassPropertyDescriptor, func: Callable[..., Any]) -> None:
            cls_: Any = type(instance)

            class _(cls_):  # noqa: N801
                @func
                def __get__(self, obj: Any, cls: Any = None) -> Any:
                    return super().__get__(obj, cls)

            instance.__class__ = _

        # Add `deprecated` decorator on the top of `__get__`
        _patch(func, deprecated(name, version, message, stacklevel))
        return func
    return decorator


def lazy_import(module_name: str) -> ModuleType:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Module {module_name} not found")

    loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to load module {module_name}") from e
    return module
