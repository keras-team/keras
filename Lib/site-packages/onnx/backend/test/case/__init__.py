# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

from onnx.backend.test.case.base import Snippets
from onnx.backend.test.case.utils import import_recursive


def collect_snippets() -> dict[str, list[tuple[str, str]]]:
    import_recursive(sys.modules[__name__])
    return Snippets
