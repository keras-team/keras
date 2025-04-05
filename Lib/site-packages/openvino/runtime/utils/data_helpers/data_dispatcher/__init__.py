# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.utils.data_helpers.data_dispatcher import ContainerTypes
from openvino.utils.data_helpers.data_dispatcher import ScalarTypes
from openvino.utils.data_helpers.data_dispatcher import ValidKeys

from openvino.utils.data_helpers.data_dispatcher import is_list_simple_type
from openvino.utils.data_helpers.data_dispatcher import get_request_tensor
from openvino.utils.data_helpers.data_dispatcher import value_to_tensor
from openvino.utils.data_helpers.data_dispatcher import to_c_style
from openvino.utils.data_helpers.data_dispatcher import normalize_arrays
from openvino.utils.data_helpers.data_dispatcher import create_shared
from openvino.utils.data_helpers.data_dispatcher import set_request_tensor
from openvino.utils.data_helpers.data_dispatcher import update_tensor
from openvino.utils.data_helpers.data_dispatcher import update_inputs
from openvino.utils.data_helpers.data_dispatcher import create_copied
from openvino.utils.data_helpers.data_dispatcher import _data_dispatch
