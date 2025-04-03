# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties import CacheMode
from openvino._pyopenvino.properties import WorkloadType

# Properties
import openvino._pyopenvino.properties as __properties
from openvino.properties._properties import __make_properties
__make_properties(__properties, __name__)

# Submodules
from openvino.properties import hint
from openvino.properties import intel_cpu
from openvino.properties import intel_gpu
from openvino.properties import intel_auto
from openvino.properties import device
from openvino.properties import log
from openvino.properties import streams
