# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Enums
from openvino._pyopenvino.properties.log import Level

# Properties
import openvino._pyopenvino.properties.log as __log
from openvino.properties._properties import __make_properties
__make_properties(__log, __name__)
