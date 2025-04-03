# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

from openvino.tools.ovc.error import Error


def default_path():
    EXT_DIR_NAME = '.'
    return os.path.abspath(os.getcwd().join(EXT_DIR_NAME))


def any_extensions_used(argv: argparse.Namespace):
    # Checks that extensions are provided.
    # Allowed types are string containing path to legacy extension directory
    # or path to new extension .so file, or classes inherited from BaseExtension.
    if not hasattr(argv, 'extension') or argv.extension is None:
        return False
    if not isinstance(argv.extension, (list, tuple)):
        argv.extension = [argv.extension]

    if isinstance(argv.extension, (list, tuple)) and len(argv.extension) > 0:
        has_non_default_path = False
        has_non_str_objects = False
        for ext in argv.extension:
            if not isinstance(ext, str):
                has_non_str_objects = True
                continue
            if len(ext) == 0 or ext == default_path():
                continue
            has_non_default_path = True

        return has_non_default_path or has_non_str_objects

    raise Exception("Expected list of extensions, got {}.".format(type(argv.extension)))


def get_transformations_config_path(argv: argparse.Namespace) -> Path:
    if hasattr(argv, 'transformations_config') \
            and argv.transformations_config is not None and len(argv.transformations_config):
        if isinstance(argv.transformations_config, str):
            path = Path(argv.transformations_config)
            if path.is_file():
                return path
    return None


def legacy_transformations_config_used(argv: argparse.Namespace):
    return get_transformations_config_path(argv) != None


def tensorflow_custom_operations_config_update_used(argv: argparse.Namespace):
    return hasattr(argv, 'tensorflow_custom_operations_config_update') and \
           argv.tensorflow_custom_operations_config_update is not None
