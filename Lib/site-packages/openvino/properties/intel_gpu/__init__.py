# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Properties
import openvino._pyopenvino.properties.intel_gpu as __intel_gpu
from openvino.properties._properties import __make_properties
__make_properties(__intel_gpu, __name__)

# Classes
from openvino._pyopenvino.properties.intel_gpu import MemoryType
from openvino._pyopenvino.properties.intel_gpu import CapabilityGPU
