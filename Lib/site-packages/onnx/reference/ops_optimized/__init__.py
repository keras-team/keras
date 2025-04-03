# Copyright (c) ONNX Project Contributors

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.ops_optimized.op_conv_optimized import Conv

optimized_operators = [Conv]

__all__ = ["Conv", "optimized_operators"]
