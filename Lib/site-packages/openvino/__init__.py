# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

# Required for Windows OS platforms
# Note: always top-level
try:
    from openvino.package_utils import _add_openvino_libs_to_search_path
    _add_openvino_libs_to_search_path()
except ImportError:
    pass

# #
# # OpenVINO API
# # This __init__.py forces checking of runtime modules to propagate errors.
# # It is not compared with init files from openvino-dev package.
# #

# Openvino pybind bindings
from openvino._pyopenvino import AxisSet
from openvino._pyopenvino import AxisVector
from openvino._pyopenvino import ConstOutput
from openvino._pyopenvino import Coordinate
from openvino._pyopenvino import CoordinateDiff
from openvino._pyopenvino import DiscreteTypeInfo
from openvino._pyopenvino import Extension
from openvino._pyopenvino import ProfilingInfo
from openvino._pyopenvino import RTMap
from openvino._pyopenvino import Version
from openvino._pyopenvino import Symbol
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import Input
from openvino._pyopenvino import OpExtension
from openvino._pyopenvino import Output
from openvino._pyopenvino import Node
from openvino._pyopenvino import Strides
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Layout
from openvino._pyopenvino import Type
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import OVAny
from openvino._pyopenvino import get_batch
from openvino._pyopenvino import set_batch
from openvino._pyopenvino import serialize
from openvino._pyopenvino import shutdown
from openvino._pyopenvino import save_model
from openvino._pyopenvino import layout_helpers
from openvino._pyopenvino import RemoteContext
from openvino._pyopenvino import RemoteTensor

# Import public classes from _ov_api
from openvino._ov_api import Model
from openvino._ov_api import Core
from openvino._ov_api import CompiledModel
from openvino._ov_api import InferRequest
from openvino._ov_api import AsyncInferQueue
from openvino._ov_api import Op

# Import all public modules
from openvino.package_utils import lazy_import
runtime = lazy_import("openvino.runtime")
from openvino import frontend as frontend
from openvino import helpers as helpers
from openvino import experimental as experimental
from openvino import preprocess as preprocess
from openvino import utils as utils
from openvino import properties as properties

# Helper functions for openvino module
from openvino.utils.data_helpers import tensor_from_file
from openvino._ov_api import compile_model


# Import opsets
from openvino import op
from openvino import opset1
from openvino import opset2
from openvino import opset3
from openvino import opset4
from openvino import opset5
from openvino import opset6
from openvino import opset7
from openvino import opset8
from openvino import opset9
from openvino import opset10
from openvino import opset11
from openvino import opset12
from openvino import opset13
from openvino import opset14
from openvino import opset15
from openvino import opset16

# libva related:
from openvino._pyopenvino import VAContext
from openvino._pyopenvino import VASurfaceTensor

# Set version for openvino package
from openvino._pyopenvino import get_version
__version__ = get_version()

# Tools
try:
    # Model Conversion API - ovc should reside in the main namespace
    from openvino.tools.ovc import convert_model
except ImportError:
    pass
