# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx
from onnx import parser, printer


class TestBasicFunctions(unittest.TestCase):
    def check_graph(self, graph: onnx.GraphProto) -> None:
        self.assertEqual(len(graph.node), 3)
        self.assertEqual(graph.node[0].op_type, "MatMul")
        self.assertEqual(graph.node[1].op_type, "Add")
        self.assertEqual(graph.node[2].op_type, "Softmax")

    def test_parse_graph(self) -> None:
        text0 = """
           agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
           {
              T = MatMul(X, W)
              S = Add(T, B)
              C = Softmax(S)
           }
           """
        graph1 = parser.parse_graph(text0)
        text1 = printer.to_text(graph1)
        graph2 = parser.parse_graph(text1)
        text2 = printer.to_text(graph2)
        # Note that text0 and text1 should be semantically-equivalent, but may differ
        # in white-space and other syntactic sugar. However, we expect text1 and text2
        # to be identical.
        self.assertEqual(text1, text2)
        self.check_graph(graph2)


if __name__ == "__main__":
    unittest.main()
