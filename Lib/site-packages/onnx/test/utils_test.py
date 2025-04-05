# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper


class TestUtilityFunctions(unittest.TestCase):
    def test_extract_model(self) -> None:
        def create_tensor(name):  # type: ignore
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

        A0 = create_tensor("A0")
        A1 = create_tensor("A1")
        B0 = create_tensor("B0")
        B1 = create_tensor("B1")
        B2 = create_tensor("B2")
        C0 = create_tensor("C0")
        C1 = create_tensor("C1")
        D0 = create_tensor("D0")
        L0_0 = helper.make_node("Add", ["A0", "A1"], ["B0"])
        L0_1 = helper.make_node("Sub", ["A0", "A1"], ["B1"])
        L0_2 = helper.make_node("Mul", ["A0", "A1"], ["B2"])
        L1_0 = helper.make_node("Add", ["B0", "B1"], ["C0"])
        L1_1 = helper.make_node("Sub", ["B1", "B2"], ["C1"])
        L2_0 = helper.make_node("Mul", ["C0", "C1"], ["D0"])

        g0 = helper.make_graph(
            [L0_0, L0_1, L0_2, L1_0, L1_1, L2_0], "test", [A0, A1], [D0]
        )
        m0 = helper.make_model(g0, producer_name="test")
        tdir = tempfile.mkdtemp()
        p0 = os.path.join(tdir, "original.onnx")
        onnx.save(m0, p0)

        p1 = os.path.join(tdir, "extracted.onnx")
        input_names = ["B0", "B1", "B2"]
        output_names = ["C0", "C1"]
        onnx.utils.extract_model(p0, p1, input_names, output_names)

        m1 = onnx.load(p1)
        self.assertEqual(m1.producer_name, "onnx.utils.extract_model")
        self.assertEqual(m1.ir_version, m0.ir_version)
        self.assertEqual(m1.opset_import, m0.opset_import)
        self.assertEqual(len(m1.graph.node), 2)
        self.assertEqual(len(m1.graph.input), 3)
        self.assertEqual(len(m1.graph.output), 2)
        self.assertEqual(m1.graph.input[0], B0)
        self.assertEqual(m1.graph.input[1], B1)
        self.assertEqual(m1.graph.input[2], B2)
        self.assertEqual(m1.graph.output[0], C0)
        self.assertEqual(m1.graph.output[1], C1)
        shutil.rmtree(tdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
