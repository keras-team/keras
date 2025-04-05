# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

from openvino._pyopenvino import get_version

__version__ = get_version()

# main classes
from openvino.frontend.frontend import FrontEndManager
from openvino.frontend.frontend import FrontEnd
from openvino._pyopenvino import InputModel
from openvino._pyopenvino import NodeContext
from openvino._pyopenvino import Place

# extensions
from openvino._pyopenvino import DecoderTransformationExtension
from openvino._pyopenvino import ConversionExtension
from openvino._pyopenvino.frontend import OpExtension
from openvino._pyopenvino import ProgressReporterExtension
from openvino._pyopenvino import TelemetryExtension

# exceptions
from openvino._pyopenvino import NotImplementedFailure
from openvino._pyopenvino import InitializationFailure
from openvino._pyopenvino import OpConversionFailure
from openvino._pyopenvino import OpValidationFailure
from openvino._pyopenvino import GeneralFailure
