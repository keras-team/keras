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


class QuantizeLinear(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = np.uint8(128)
        y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear",
        )

    @staticmethod
    def export_axis() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array(
            [
                [
                    [[-162, 10], [-100, 232], [-20, -50]],
                    [[-76, 0], [0, 252], [32, -44]],
                    [[245, -485], [-960, -270], [-375, -470]],
                ],
            ],
            dtype=np.float32,
        )
        y_scale = np.array([2, 4, 5], dtype=np.float32)
        y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
        y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(
            np.uint8
        )

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_axis",
        )

    @staticmethod
    def export_e4m3fn() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array([0.0, 1.0, 2.0, 100000.0, 200.0]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = make_tensor("y_zero_point", TensorProto.FLOAT8E4M3FN, [1], [0])
        y = make_tensor("y", TensorProto.FLOAT8E4M3FN, [5], [0, 0.5, 1, 448, 96])

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_e4m3fn",
        )

    @staticmethod
    def export_e5m2() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array([0.0, 1.0, 2.0, 100000.0, 200.0]).astype(np.float32)
        y_scale = np.float32(2)
        y_zero_point = make_tensor("y_zero_point", TensorProto.FLOAT8E5M2, [1], [0.0])
        y = make_tensor("y", TensorProto.FLOAT8E5M2, [5], [0, 0.5, 1, 49152, 96])

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_e5m2",
        )

    @staticmethod
    def export_uint16() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array(
            [
                0.0,
                -128.0,
                3.0,
                -3.0,
                2.9,
                -2.9,
                3.1,
                -3.1,
                65536.0,
                -65534.0,
                70000.0,
                -70000.0,
            ]
        ).astype(np.float32)
        y_scale = np.float32(2.0)
        y_zero_point = np.uint16(32767)
        y = np.array(
            [
                32767,
                32703,
                32769,
                32765,
                32768,
                32766,
                32769,
                32765,
                65535,
                0,
                65535,
                0,
            ]
        ).astype(np.uint16)

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_uint16",
        )

    @staticmethod
    def export_int16() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
        )

        x = np.array(
            [
                0.0,
                -514.0,
                3.0,
                -3.0,
                2.9,
                -2.9,
                3.1,
                -3.1,
                65022.0,
                -66046.0,
                65023.0,
                -66047.0,
                65024.0,
                -66048.0,
                70000.0,
                -70000.0,
            ]
        ).astype(np.float32)
        y_scale = np.float32(2.0)
        y_zero_point = np.int16(256)
        y = np.array(
            [
                256,
                -1,
                258,
                254,
                257,
                255,
                258,
                254,
                32767,
                -32767,
                32767,
                -32768,
                32767,
                -32768,
                32767,
                -32768,
            ]
        ).astype(np.int16)

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_int16",
        )

    @staticmethod
    def export_uint4() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
            axis=0,
        )

        x = np.array(
            [
                [0.0, 2.5, 4.8, 8.6],
                [-30, -20, 6, 9],
                [12, 15, 16, 40],
            ]
        ).astype(np.float32)

        y_scale = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
        y_zero_point = make_tensor(
            "y_zero_point", TensorProto.UINT4, y_scale.shape, np.ones_like(y_scale)
        )
        y = make_tensor(
            "y", TensorProto.UINT4, x.shape, [1, 2, 3, 5, -1, -1, 3, 4, 4, 5, 5, 11]
        )

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_uint4",
        )

    @staticmethod
    def export_int4() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
            axis=0,
        )

        x = np.array(
            [
                [0.0, 2.5, 4.8, 8.6],
                [-30, -20, 6, 9],
                [12, 15, 16, 40],
            ]
        ).astype(np.float32)

        y_scale = np.asarray([2.0, 3.0, 4.0], dtype=np.float32)
        y_zero_point = make_tensor(
            "y_zero_point", TensorProto.INT4, y_scale.shape, np.ones_like(y_scale)
        )
        y = make_tensor(
            "y", TensorProto.INT4, x.shape, [1, 2, 3, 5, -8, -6, 3, 4, 4, 5, 5, 7]
        )

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_int4",
        )

    @staticmethod
    def export_blocked_asymmetric() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale", "y_zero_point"],
            outputs=["y"],
            axis=1,
            block_size=2,
        )

        x = np.array(
            [
                [6.0, 12.0, 50.0, 5.0],
                [1.0, 8.0, 4.0, 5.0],
                [0.0, 20.0, 10.0, 4.0],
            ],
            dtype=np.float32,
        )
        y_scale = np.array(
            [
                [1.5, 2.5],
                [3.0, 4.9],
                [5.1, 6.9],
            ],
            dtype=np.float32,
        )
        y_zero_point = np.array(
            [
                [0, 1],
                [1, 0],
                [2, 3],
            ],
            dtype=np.uint8,
        )
        # x.shape = (3, 4)
        # y_scale.shape = (3, 2)
        assert y_scale.shape == y_zero_point.shape
        block_axis = 1
        # The block shape is [x.shape[i] // y_scale.shape[i] for i in range(len(x.shape))] = (1, 2)
        assert all(
            x.shape[i] == y_scale.shape[i]
            for i in range(len(x.shape))
            if i != block_axis
        )
        assert x.shape[block_axis] % y_scale.shape[block_axis] == 0
        repeats = x.shape[block_axis] // y_scale.shape[block_axis]

        # Create element-wise scale and zero point
        y_scale_elementwise = np.repeat(y_scale, repeats=repeats, axis=block_axis)
        y_zero_point_elementwise = np.repeat(
            y_zero_point, repeats=repeats, axis=block_axis
        )

        y = np.rint(x / y_scale_elementwise + y_zero_point_elementwise).astype(np.uint8)

        expect(
            node,
            inputs=[x, y_scale, y_zero_point],
            outputs=[y],
            name="test_quantizelinear_blocked_asymmetric",
        )

    @staticmethod
    def export_blocked_symmetric() -> None:
        node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=["x", "y_scale"],
            outputs=["y"],
            axis=1,
            block_size=2,
            output_dtype=TensorProto.INT16,
        )

        x = np.array(
            [
                [6.0, -8, -10, 5.0],
                [1.0, 8.0, 4.0, 5.0],
                [0.0, 20.0, 10.0, 4.0],
            ],
            dtype=np.float32,
        )

        y_scale = np.array(
            [
                [1.5, 2.5],
                [3.0, 4.9],
                [5.1, 6.9],
            ],
            dtype=np.float32,
        )

        # x.shape = (3, 4)
        # y_scale.shape = (3, 2)

        block_axis = 1
        # The block shape is [x.shape[i] // y_scale.shape[i] for i in range(len(x.shape))] = (1, 2)
        assert all(
            x.shape[i] == y_scale.shape[i]
            for i in range(len(x.shape))
            if i != block_axis
        )
        assert x.shape[block_axis] % y_scale.shape[block_axis] == 0
        repeats = x.shape[block_axis] // y_scale.shape[block_axis]

        # Create element-wise scale and zero point
        y_scale_elementwise = np.repeat(y_scale, repeats=repeats, axis=block_axis)

        y_val = np.clip(
            np.rint(x / y_scale_elementwise), a_min=-32768, a_max=32767
        ).astype(np.int16)
        y = make_tensor(
            "y",
            TensorProto.INT16,
            x.shape,
            y_val,
        )
        expect(
            node,
            inputs=[x, y_scale],
            outputs=[y],
            name="test_quantizelinear_blocked_symmetric",
        )
