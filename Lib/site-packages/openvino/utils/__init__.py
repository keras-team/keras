# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generic utilities. Factor related functions out to separate files."""

from openvino._pyopenvino.util import numpy_to_c, replace_node, replace_output_update_name

from openvino.package_utils import get_cmake_path
from openvino.package_utils import deprecated
from openvino.package_utils import classproperty
from openvino.package_utils import deprecatedclassproperty
