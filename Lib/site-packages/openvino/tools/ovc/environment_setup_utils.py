# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

# do not print INFO and WARNING messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_imported_module_version(imported_module):
    """
    Get imported module version
    :return: version(str) or raise AttributeError exception
    """
    version_attrs = ("__version__", "VERSION", "version")
    installed_version = None
    for attr in version_attrs:
        installed_version = getattr(imported_module, attr, None)
        if isinstance(installed_version, str):
            return installed_version
        else:
            installed_version = None

    if installed_version is None:
        raise AttributeError("{} module doesn't have version attribute".format(imported_module))
    else:
        return installed_version


def get_environment_setup(framework):
    """
    Get environment setup such as Python version, TensorFlow version
    :param framework: framework name
    :return: a dictionary of environment variables
    """
    env_setup = dict()
    python_version = "{}.{}.{}".format(sys.version_info.major,
                                       sys.version_info.minor,
                                       sys.version_info.micro)
    env_setup['python_version'] = python_version
    try:
        if framework == 'tf':
            exec("import tensorflow")
            env_setup['tensorflow'] = get_imported_module_version(sys.modules["tensorflow"])
            exec("del tensorflow")
    except (AttributeError, ImportError):
        pass
    env_setup['sys_platform'] = sys.platform
    return env_setup
