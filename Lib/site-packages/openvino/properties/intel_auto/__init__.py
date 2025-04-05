# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties.intel_auto import SchedulePolicy

# Properties
import openvino._pyopenvino.properties.intel_auto as __intel_auto
from openvino.properties._properties import __make_properties

__make_properties(__intel_auto, __name__)
