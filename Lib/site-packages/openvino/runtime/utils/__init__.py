# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino.utils import numpy_to_c, replace_node, replace_output_update_name

# Import runtime proxy modules for backward compatibility
from openvino.runtime.utils import broadcasting
from openvino.runtime.utils import decorators
from openvino.runtime.utils import data_helpers
from openvino.runtime.utils import input_validation
from openvino.runtime.utils import node_factory
from openvino.runtime.utils import reduction
from openvino.runtime.utils import types
