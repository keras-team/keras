# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class Gelu(Base):
    @staticmethod
    def export_gelu_tanh() -> None:
        node = onnx.helper.make_node(
            "Gelu", inputs=["x"], outputs=["y"], approximate="tanh"
        )

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-0.158808, 0., 0.841192]
        y = (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_gelu_tanh_1")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        # expected output [2.9963627, 3.99993, 4.9999995]
        y = (
            0.5
            * x
            * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
        ).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_gelu_tanh_2")

    @staticmethod
    def export_gelu_default() -> None:
        node = onnx.helper.make_node("Gelu", inputs=["x"], outputs=["y"])

        x = np.array([-1, 0, 1]).astype(np.float32)
        # expected output [-0.15865526, 0., 0.84134474]
        y = (0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_gelu_default_1")

        x = np.random.randn(3, 4, 5).astype(np.float32)
        # expected output [2.99595031, 3.99987331, 4.99999857]
        y = (0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2)))).astype(np.float32)
        expect(node, inputs=[x], outputs=[y], name="test_gelu_default_2")
