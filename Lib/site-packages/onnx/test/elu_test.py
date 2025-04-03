# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from onnx import checker, defs, helper


class TestRelu(unittest.TestCase):
    def test_elu(self) -> None:
        self.assertTrue(defs.has("Elu"))
        node_def = helper.make_node("Elu", ["X"], ["Y"], alpha=1.0)
        checker.check_node(node_def)


if __name__ == "__main__":
    unittest.main()
