# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

from openvino.tools.ovc.main import main
from openvino.tools.ovc.telemetry_utils import init_ovc_telemetry

init_ovc_telemetry()
sys.exit(main())
