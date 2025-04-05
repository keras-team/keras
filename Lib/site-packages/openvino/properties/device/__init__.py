# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties.device import Type

# Properties
import openvino._pyopenvino.properties.device as __device
from openvino.properties._properties import __make_properties
__make_properties(__device, __name__)

# Classes
from openvino._pyopenvino.properties.device import Capability
