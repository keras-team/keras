# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx import TensorProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.helper import make_tensor


class DequantizeLinear(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
        )

        # scalar zero point and scale
        x = np.array([0, 3, 128, 255]).astype(np.uint8)
        x_scale = np.float32(2)
        x_zero_point = np.uint8(128)
        y = np.array([-256, -250, 0, 254], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear",
        )

    @staticmethod
    def export_axis() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
        )

        # 1-D tensor zero point and scale of size equal to axis 1 of the input tensor
        x = np.array(
            [
                [
                    [[3, 89], [34, 200], [74, 59]],
                    [[5, 24], [24, 87], [32, 13]],
                    [[245, 99], [4, 142], [121, 102]],
                ],
            ],
            dtype=np.uint8,
        )
        x_scale = np.array([2, 4, 5], dtype=np.float32)
        x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
        y = (
            x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)
        ) * x_scale.reshape(1, 3, 1, 1)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_axis",
        )

    @staticmethod
    def export_e4m3fn() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
        x_scale = np.float32(2)
        y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale],
            outputs=[y],
            name="test_dequantizelinear_e4m3fn",
        )

    @staticmethod
    def export_e4m3fn_float16() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
        x_scale = np.float16(2)
        y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float16)

        expect(
            node,
            inputs=[x, x_scale],
            outputs=[y],
            name="test_dequantizelinear_e4m3fn_float16",
        )

    @staticmethod
    def export_e4m3fn_zero_point() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "zero_point"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, -104])
        zero_point = make_tensor("zero_point", TensorProto.FLOAT8E4M3FN, [1], [0])
        x_scale = np.float32(2)
        y = np.array([0.0, 1.0, 2.0, 896.0, -208.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, zero_point],
            outputs=[y],
            name="test_dequantizelinear_e4m3fn_zero_point",
        )

    @staticmethod
    def export_e5m2() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.FLOAT8E5M2, [5], [0, 0.5, 1, 49152, -96])
        x_scale = np.float32(2)
        y = np.array([0.0, 1.0, 2.0, 98304.0, -192.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale],
            outputs=[y],
            name="test_dequantizelinear_e5m2",
        )

    @staticmethod
    def export_uint16() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
        )

        x = np.array([30000, 31000, 32768, 33000]).astype(np.uint16)
        x_scale = np.float32(2)
        x_zero_point = np.uint16(32767)
        y = np.array([-5534.0, -3534.0, 2.0, 466.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_uint16",
        )

    @staticmethod
    def export_int16() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
        )

        x = np.array([-300, -30, -1025, 1270]).astype(np.int16)
        x_scale = np.float32(2)
        x_zero_point = np.int16(-1024)
        y = np.array([1448.0, 1988.0, -2.0, 4588.0], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_int16",
        )

    @staticmethod
    def export_uint4() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.UINT4, [5], [0, 1, 7, 10, 15])
        x_scale = np.float32(2)
        x_zero_point = make_tensor("x_zero_point", TensorProto.UINT4, (1,), [1])
        y = np.array([-2, 0, 12, 18, 28], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_uint4",
        )

    @staticmethod
    def export_int4() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
            axis=0,
        )

        # scalar zero point and scale
        x = make_tensor("x", TensorProto.INT4, [5], [0, 1, 7, -4, -8])
        x_scale = np.float32(2)
        x_zero_point = make_tensor("x_zero_point", TensorProto.INT4, (1,), [1])
        y = np.array([-2, 0, 12, -10, -18], dtype=np.float32)

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_int4",
        )

    @staticmethod
    def export_blocked() -> None:
        node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=["x", "x_scale", "x_zero_point"],
            outputs=["y"],
            axis=1,
            block_size=2,
        )

        x = np.array(
            [
                [
                    [[3, 89], [34, 200], [74, 59]],
                    [[5, 24], [24, 87], [32, 13]],
                    [[5, 12], [12, 33], [65, 42]],
                    [[245, 99], [4, 142], [121, 102]],
                ],
            ],
            dtype=np.uint8,
        )

        x_scale = np.array(
            [
                [
                    [[3.0, 2.0], [4.0, 1.0], [2.0, 2.0]],
                    [[5.0, 2.0], [4.0, 3.0], [5.0, 2.0]],
                ],
            ],
            dtype=np.float32,
        )
        x_zero_point = np.array(
            [
                [
                    [[1, 0], [0, 1], [2, 20]],
                    [[3, 2], [4, 3], [15, 2]],
                ],
            ],
            dtype=np.uint8,
        )

        # x.shape = (1, 4, 3, 2)
        # x_scale.shape = (1, 2, 3, 2)
        assert x_scale.shape == x_zero_point.shape
        block_axis = 1
        # The block shape is [x.shape[i] // x_scale.shape[i] for i in range(len(x.shape))] = (1, 2, 1, 1)
        assert all(
            x.shape[i] == x_scale.shape[i]
            for i in range(len(x.shape))
            if i != block_axis
        )
        assert x.shape[block_axis] % x_scale.shape[block_axis] == 0
        repeats = x.shape[block_axis] // x_scale.shape[block_axis]

        # Create element-wise scale and zero point
        x_scale_elementwise = np.repeat(x_scale, repeats=repeats, axis=block_axis)
        x_zero_point_elementwise = np.repeat(
            x_zero_point, repeats=repeats, axis=block_axis
        )

        y = (
            x.astype(np.float32) - x_zero_point_elementwise.astype(np.float32)
        ) * x_scale_elementwise

        expect(
            node,
            inputs=[x, x_scale, x_zero_point],
            outputs=[y],
            name="test_dequantizelinear_blocked",
        )
