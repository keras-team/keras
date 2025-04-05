# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties.hint import Priority
from openvino._pyopenvino.properties.hint import SchedulingCoreType
from openvino._pyopenvino.properties.hint import ModelDistributionPolicy
from openvino._pyopenvino.properties.hint import ExecutionMode
from openvino._pyopenvino.properties.hint import PerformanceMode

# Properties
import openvino._pyopenvino.properties.hint as __hint
from openvino.properties._properties import __make_properties
__make_properties(__hint, __name__)
