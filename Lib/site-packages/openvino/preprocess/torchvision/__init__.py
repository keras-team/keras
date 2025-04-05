# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Torchvision to OpenVINO preprocess converter.
"""

# flake8: noqa

from openvino._pyopenvino import get_version as _get_version

__version__ = _get_version()

from .preprocess_converter import PreprocessConverter
