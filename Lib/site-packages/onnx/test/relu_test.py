# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

from onnx import defs, helper


class TestRelu(unittest.TestCase):
    def test_relu(self) -> None:
        self.assertTrue(defs.has("Relu"))
        helper.make_node("Relu", ["X"], ["Y"])


if __name__ == "__main__":
    unittest.main()
