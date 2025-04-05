# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import automatic_conversion_test_base
import numpy as np

import onnx
from onnx import TensorProto, helper

#####################################################################################
# Every test calls _test_op_conversion to upgrade a model from an initial opset version
# to the most recent version and runs checker and shape inference on the final upgraded model.
####################################################################################


class TestAutomaticUpgrade(automatic_conversion_test_base.TestAutomaticConversion):
    @classmethod
    def setUpClass(cls):
        cls.tested_ops = []

    def _test_op_upgrade(self, op, *args, **kwargs):
        self.tested_ops.append(op)
        self._test_op_conversion(op, *args, **kwargs, is_upgrade=True)

    def test_Abs(self) -> None:
        self._test_op_upgrade("Abs", 1, attrs={"consumed_inputs": [0]})

    def test_Acosh(self) -> None:
        self._test_op_upgrade("Acosh", 9)

    def test_Acos(self) -> None:
        self._test_op_upgrade("Acos", 7)

    def test_And(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "And",
            7,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.BOOL, TensorProto.BOOL],
            [TensorProto.BOOL],
        )

    def test_Asinh(self) -> None:
        self._test_op_upgrade("Asinh", 9)

    def test_Atanh(self) -> None:
        self._test_op_upgrade("Atanh", 9)

    def test_Add_1(self) -> None:
        self._test_op_upgrade(
            "Add", 1, [[3, 4, 5], [3, 4, 5]], attrs={"consumed_inputs": [0]}
        )

    def test_Add_2(self) -> None:
        self._test_op_upgrade(
            "Add", 1, [[3, 4, 5], [5]], attrs={"consumed_inputs": [0], "broadcast": 1}
        )

    def test_Add_3(self) -> None:
        self._test_op_upgrade(
            "Add",
            1,
            [[3, 4, 5], [3]],
            attrs={"consumed_inputs": [0], "broadcast": 1, "axis": 0},
        )

    def test_AffineGrid_2D(self) -> None:
        N, _, H, W = 2, 3, 5, 6
        self._test_op_upgrade("AffineGrid", 20, [[N, 2, 3], [4]], [[N, H, W, 2]])

    def test_AffineGrid_3D(self) -> None:
        N, _, D, H, W = 2, 3, 4, 5, 6
        self._test_op_upgrade("AffineGrid", 20, [[N, 3, 4], [5]], [[N, D, H, W, 3]])

    def test_ArgMax_1(self) -> None:
        self._test_op_upgrade(
            "ArgMax", 7, [[2, 3, 4]], [[1, 3, 4]], output_types=[TensorProto.INT64]
        )

    def test_ArgMax_2(self) -> None:
        self._test_op_upgrade(
            "ArgMax",
            7,
            [[2, 3, 4]],
            [[2, 1, 4]],
            output_types=[TensorProto.INT64],
            attrs={"axis": 1},
        )

    def test_ArgMin_1(self) -> None:
        self._test_op_upgrade(
            "ArgMin", 7, [[2, 3, 4]], [[1, 3, 4]], output_types=[TensorProto.INT64]
        )

    def test_ArgMin_2(self) -> None:
        self._test_op_upgrade(
            "ArgMin",
            7,
            [[2, 3, 4]],
            [[2, 1, 4]],
            output_types=[TensorProto.INT64],
            attrs={"axis": 1},
        )

    def test_Asin(self) -> None:
        self._test_op_upgrade("Asin", 7)

    def test_Atan(self) -> None:
        self._test_op_upgrade("Atan", 7)

    def test_AveragePool(self) -> None:
        self._test_op_upgrade(
            "AveragePool",
            1,
            [[1, 1, 5, 5]],
            [[1, 1, 4, 4]],
            attrs={"kernel_shape": [2, 2]},
        )

    def test_Bernoulli(self) -> None:
        self._test_op_upgrade("Bernoulli", 15)

    def test_BitShift(self) -> None:
        self._test_op_upgrade(
            "BitShift",
            11,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.UINT8, TensorProto.UINT8],
            [TensorProto.UINT8],
            attrs={"direction": "RIGHT"},
        )

    def test_BatchNormalization_1(self) -> None:
        self._test_op_upgrade(
            "BatchNormalization",
            1,
            [[1, 3], [3], [3], [3], [3]],
            [[1, 3]],
            attrs={"consumed_inputs": [1, 1], "is_test": 1, "spatial": 1},
        )

    def test_BatchNormalization_2(self) -> None:
        self._test_op_upgrade(
            "BatchNormalization",
            14,
            [[1, 3], [3], [3], [3], [3]],
            [[1, 3], [3], [3]],
            attrs={"training_mode": 1},
        )

    def test_Cast(self) -> None:
        # 5->6 adapter is missing
        self._test_op_upgrade(
            "Cast", 6, [[2, 3]], [[2, 3]], [TensorProto.INT64], attrs={"to": 1}
        )

    def test_Ceil(self) -> None:
        self._test_op_upgrade("Ceil", 1, attrs={"consumed_inputs": [0]})

    def test_Celu(self) -> None:
        self._test_op_upgrade("Celu", 12)

    def test_Clip_1(self) -> None:
        self._test_op_upgrade("Clip", 1, attrs={"consumed_inputs": [0]})

    def test_Clip_2(self) -> None:
        self._test_op_upgrade("Clip", 1, attrs={"consumed_inputs": [0], "min": -1.4})

    def test_Clip_3(self) -> None:
        self._test_op_upgrade("Clip", 1, attrs={"consumed_inputs": [0], "max": 2.6})

    def test_Clip_4(self) -> None:
        self._test_op_upgrade(
            "Clip", 1, attrs={"consumed_inputs": [0], "min": -1.4, "max": 2.6}
        )

    def test_Col2Im_4D(self) -> None:
        self._test_op_upgrade("Col2Im", 18, [[1, 5, 5], [2], [2]], [[1, 1, 5, 5]])

    def test_Col2Im_5D(self) -> None:
        self._test_op_upgrade("Col2Im", 18, [[1, 10, 12], [3], [3]], [[1, 2, 3, 4, 5]])

    def test_Compress(self) -> None:
        self._test_op_upgrade(
            "Compress",
            9,
            [[6, 7], [3]],
            [[3]],
            [TensorProto.FLOAT, TensorProto.BOOL],
            [TensorProto.FLOAT],
        )

    def test_Concat(self) -> None:
        self._test_op_upgrade("Concat", 1, [[2, 3], [2, 4]], [[2, 7]])

    def test_constant(self) -> None:
        value = helper.make_tensor(
            "Value",
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(),
            raw=True,
        )
        self._test_op_upgrade("Constant", 1, [], attrs={"value": value})

    def test_ConstantOfShape(self) -> None:
        self._test_op_upgrade("ConstantOfShape", 9, [[3]])

    def test_Conv_1(self) -> None:
        self._test_op_upgrade(
            "Conv", 1, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]]
        )

    def test_Conv_2(self) -> None:
        self._test_op_upgrade(
            "Conv", 1, [[1, 3, 5, 5], [4, 3, 2, 2], [4]], [[1, 4, 4, 4]]
        )

    def test_Conv_3(self) -> None:
        self._test_op_upgrade(
            "Conv",
            1,
            [[1, 3, 5, 5], [4, 1, 2, 2], [4]],
            [[1, 4, 3, 7]],
            attrs={
                "dilations": [1, 2],
                "group": 3,
                "pads": [0, 1, 2, 3],
                "strides": [2, 1],
            },
        )

    def test_Convinteger(self) -> None:
        self._test_op_upgrade(
            "ConvInteger",
            10,
            [[1, 3, 5, 5], [4, 3, 2, 2], [4]],
            [[1, 4, 4, 4]],
            [TensorProto.UINT8, TensorProto.UINT8, TensorProto.UINT8],
            [TensorProto.INT32],
        )

    def test_ConvTranspose(self) -> None:
        self._test_op_upgrade(
            "ConvTranspose", 1, [[1, 1, 5, 5], [1, 1, 3, 3]], [[1, 1, 7, 7]]
        )

    def test_DeformConv(self) -> None:
        self._test_op_upgrade(
            "DeformConv",
            19,
            [[1, 1, 3, 3], [1, 1, 2, 2], [1, 8, 2, 2]],
            [[1, 1, 2, 2]],
        )

    def test_Cosh(self) -> None:
        self._test_op_upgrade("Cosh", 9)

    def test_Cos(self) -> None:
        self._test_op_upgrade("Cos", 7)

    def test_Cumsum(self) -> None:
        self._test_op_upgrade(
            "CumSum",
            11,
            [[3, 4, 5], []],
            [[3, 4, 5]],
            [TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_DepthToSpace(self) -> None:
        self._test_op_upgrade(
            "DepthToSpace", 1, [[1, 8, 3, 3]], [[1, 2, 6, 6]], attrs={"blocksize": 2}
        )

    def test_DequantizeLinear(self) -> None:
        self._test_op_upgrade(
            "DequantizeLinear",
            10,
            [[2, 3], [], []],
            [[2, 3]],
            [TensorProto.INT8, TensorProto.FLOAT, TensorProto.INT8],
        )

    def test_Det_1(self) -> None:
        self._test_op_upgrade("Det", 11, [[3, 5, 5]], [[3]])

    def test_Det_2(self) -> None:
        self._test_op_upgrade("Det", 11, [[5, 5]], [[]])

    def test_DynamicQuantizeLinear(self) -> None:
        self._test_op_upgrade(
            "DynamicQuantizeLinear",
            11,
            [[3, 4, 5]],
            [[3, 4, 5], [], []],
            output_types=[TensorProto.UINT8, TensorProto.FLOAT, TensorProto.UINT8],
        )

    def test_Div(self) -> None:
        self._test_op_upgrade(
            "Div", 1, [[3, 4, 5], [3, 1, 5]], attrs={"consumed_inputs": [0]}
        )

    def test_Dropout(self) -> None:
        self._test_op_upgrade(
            "Dropout", 1, attrs={"consumed_inputs": [0], "is_test": 1}
        )

    def test_Einsum_1(self) -> None:
        self._test_op_upgrade(
            "Einsum",
            12,
            [[3, 4, 5], [3, 5, 6]],
            [[3, 4, 6]],
            attrs={"equation": "bij, bjk -> bik"},
        )

    def test_Einsum_2(self) -> None:
        self._test_op_upgrade(
            "Einsum", 12, [[4, 5]], [[5, 4]], attrs={"equation": "ij->ji"}
        )

    def test_Elu(self) -> None:
        self._test_op_upgrade("Elu", 1, attrs={"consumed_inputs": [0]})

    def test_Equal(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "Equal", 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL]
        )

    def test_Erf(self) -> None:
        self._test_op_upgrade("Erf", 9)

    def test_Exp(self) -> None:
        self._test_op_upgrade("Exp", 1, attrs={"consumed_inputs": [0]})

    def test_Expand(self) -> None:
        shape = helper.make_tensor(
            "b", TensorProto.INT64, dims=[4], vals=np.array([5, 2, 6, 4])
        )
        self._test_op_upgrade(
            "Expand",
            8,
            [[2, 1, 4], [4]],
            [[5, 2, 6, 4]],
            [TensorProto.FLOAT, TensorProto.INT64],
            initializer=[shape],
        )

    def test_EyeLike(self) -> None:
        self._test_op_upgrade("EyeLike", 9, [[4, 5]], [[4, 5]])

    def test_Flatten(self) -> None:
        self._test_op_upgrade("Flatten", 1, [[3, 4, 5]], [[3, 20]], attrs={"axis": 1})

    def test_Floor(self) -> None:
        self._test_op_upgrade("Floor", 1, attrs={"consumed_inputs": [0]})

    def test_Gather(self) -> None:
        self._test_op_upgrade(
            "Gather",
            1,
            [[3, 4, 5], [6, 7]],
            [[6, 7, 4, 5]],
            [TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_GatherElements(self) -> None:
        self._test_op_upgrade(
            "GatherElements",
            11,
            [[3, 4, 5], [6, 7]],
            [[6, 7]],
            [TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_GatherND(self) -> None:
        self._test_op_upgrade("GatherND", 11, [[1, 2, 3], [1, 2, 3]], [[1, 2]])

    def test_Gelu_approximate_tanh(self) -> None:
        self._test_op_upgrade("Gelu", 20, attrs={"approximate": "tanh"})

    def test_Gelu(self) -> None:
        self._test_op_upgrade("Gelu", 20)

    def test_Gemm(self) -> None:
        self._test_op_upgrade("Gemm", 1, [[5, 4], [4, 3], [3]], [[5, 3]])

    def test_GlobalAveragePool(self) -> None:
        self._test_op_upgrade("GlobalAveragePool", 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])

    def test_GlobalMaxPool(self) -> None:
        self._test_op_upgrade("GlobalMaxPool", 1, [[1, 3, 10, 10]], [[1, 3, 1, 1]])

    def test_GlobalLpPool(self) -> None:
        # 1->2 adapter is missing
        self._test_op_upgrade("GlobalLpPool", 2, [[1, 3, 10, 10]], [[1, 3, 1, 1]])

    def test_Greater(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "Greater", 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL]
        )

    def test_GreaterOrEqual(self) -> None:
        self._test_op_upgrade(
            "GreaterOrEqual",
            12,
            [[2, 3], [2, 3]],
            [[2, 3]],
            output_types=[TensorProto.BOOL],
        )

    def test_GridSample(self) -> None:
        self._test_op_upgrade(
            "GridSample",
            16,
            [[1, 1, 3, 3], [1, 3, 3, 2]],
            [[1, 1, 3, 3]],
            input_types=[TensorProto.FLOAT, TensorProto.FLOAT],
            output_types=[TensorProto.FLOAT],
            attrs={"mode": "nearest", "padding_mode": "border", "align_corners": 1},
        )

    def test_GRU_1(self) -> None:
        # 2->3, 6->7 adapters are missing
        self._test_op_upgrade(
            "GRU",
            7,
            [[5, 3, 4], [1, 18, 4], [1, 18, 4]],
            [[5, 1, 3, 6], [1, 3, 6]],
            attrs={"hidden_size": 6},
        )

    def test_GRU_2(self) -> None:
        # 2->3, 6->7 adapters are missing
        self._test_op_upgrade(
            "GRU",
            7,
            [[5, 3, 4], [2, 18, 4], [2, 18, 4]],
            [[5, 2, 3, 6], [2, 3, 6]],
            attrs={"hidden_size": 6, "direction": "bidirectional"},
        )

    def test_GRU_3(self) -> None:
        # 2->3, 6->7 adapters are missing
        self._test_op_upgrade(
            "GRU",
            7,
            [[5, 3, 4], [1, 18, 4], [1, 18, 4], [1, 24], [5], [1, 5, 6]],
            [[5, 1, 3, 6], [1, 3, 6]],
            [
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
            ],
            attrs={"hidden_size": 6},
        )

    def test_HardSigmoid(self) -> None:
        self._test_op_upgrade("HardSigmoid", 1, attrs={"consumed_inputs": [0]})

    def test_HardSwish(self) -> None:
        self._test_op_upgrade("HardSwish", 14)

    def test_Hardmax(self) -> None:
        self._test_op_upgrade("Hardmax", 1)

    def test_Identity(self) -> None:
        self._test_op_upgrade("Identity", 1)

    def test_If(self) -> None:
        sub_output = [
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [3, 4, 5])
        ]
        then_tensor = helper.make_tensor(
            "Value",
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(),
            raw=True,
        )
        then_node = helper.make_node("Constant", [], ["out"], value=then_tensor)
        then_graph = helper.make_graph([then_node], "then_graph", [], sub_output, [])
        else_tensor = helper.make_tensor(
            "Value",
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(),
            raw=True,
        )
        else_node = helper.make_node("Constant", [], ["out"], value=else_tensor)
        else_graph = helper.make_graph([else_node], "else_graph", [], sub_output, [])
        self._test_op_upgrade(
            "If",
            1,
            [[0]],
            [[3, 4, 5]],
            [TensorProto.BOOL],
            attrs={"then_branch": then_graph, "else_branch": else_graph},
        )

    def test_ImageDecoder(self) -> None:
        self._test_op_upgrade(
            "ImageDecoder",
            20,
            [[None]],
            [[None, None, 3]],
            input_types=[TensorProto.UINT8],
            output_types=[TensorProto.UINT8],
        )

    def test_InstanceNormalization(self) -> None:
        self._test_op_upgrade(
            "InstanceNormalization",
            1,
            [[1, 3], [3], [3]],
            [[1, 3]],
            attrs={"consumed_inputs": [0]},
        )

    def test_IsInf(self) -> None:
        self._test_op_upgrade(
            "IsInf", 10, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL]
        )

    def test_IsNaN(self) -> None:
        self._test_op_upgrade(
            "IsNaN", 9, [[2, 3]], [[2, 3]], output_types=[TensorProto.BOOL]
        )

    def test_LeakyRelu(self) -> None:
        self._test_op_upgrade("LeakyRelu", 1, attrs={"consumed_inputs": [0]})

    def test_Less(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "Less", 7, [[2, 3], [2, 3]], [[2, 3]], output_types=[TensorProto.BOOL]
        )

    def test_LessOrEqual(self) -> None:
        self._test_op_upgrade(
            "LessOrEqual",
            12,
            [[2, 3], [2, 3]],
            [[2, 3]],
            output_types=[TensorProto.BOOL],
        )

    def test_Log(self) -> None:
        self._test_op_upgrade("Log", 1, attrs={"consumed_inputs": [0]})

    def test_LogSoftmax(self) -> None:
        self._test_op_upgrade("LogSoftmax", 1)

    def test_Loop_1(self) -> None:
        iter_count = onnx.helper.make_tensor_value_info(
            "iter_count", onnx.TensorProto.INT64, []
        )
        cond_in = onnx.helper.make_tensor_value_info(
            "cond_in", onnx.TensorProto.BOOL, []
        )
        x_in = onnx.helper.make_tensor_value_info("x_in", onnx.TensorProto.FLOAT, [1])
        cond_out = onnx.helper.make_tensor_value_info(
            "cond_out", onnx.TensorProto.BOOL, []
        )
        x_out = onnx.helper.make_tensor_value_info("x_out", onnx.TensorProto.FLOAT, [1])
        x_scan = onnx.helper.make_tensor_value_info(
            "x_scan", onnx.TensorProto.FLOAT, [1]
        )
        const = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["one"],
            value=onnx.helper.make_tensor(
                name="value",
                data_type=onnx.TensorProto.FLOAT,
                dims=[1],
                vals=np.array([1]).astype(np.float32).astype(float),
            ),
        )
        add = onnx.helper.make_node("Add", inputs=["x_in", "one"], outputs=["x_out"])
        id_1 = onnx.helper.make_node("Identity", inputs=["x_out"], outputs=["x_scan"])
        id_2 = onnx.helper.make_node(
            "Identity", inputs=["cond_in"], outputs=["cond_out"]
        )
        loop_body = onnx.helper.make_graph(
            [const, add, id_1, id_2],
            "loop_body",
            [iter_count, cond_in, x_in],
            [cond_out, x_out, x_scan],
        )
        self._test_op_upgrade(
            "Loop",
            1,
            [[], "", [1]],
            [[1], [5, 1]],
            [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT],
            attrs={"body": loop_body},
        )

    def test_Loop_2(self) -> None:
        iter_count = onnx.helper.make_tensor_value_info(
            "iter_count", onnx.TensorProto.INT64, []
        )
        cond_in = onnx.helper.make_tensor_value_info(
            "cond_in", onnx.TensorProto.BOOL, []
        )
        x_in = onnx.helper.make_tensor_value_info(
            "x_in", onnx.TensorProto.FLOAT, [2, 1]
        )
        cond_out = onnx.helper.make_tensor_value_info(
            "cond_out", onnx.TensorProto.BOOL, []
        )
        x_out = onnx.helper.make_tensor_value_info(
            "x_out", onnx.TensorProto.FLOAT, [2, 1]
        )
        squeeze = onnx.helper.make_node(
            "Squeeze", inputs=["x_in"], outputs=["squeeze_out"], axes=[1]
        )
        unsqueeze = onnx.helper.make_node(
            "Unsqueeze", inputs=["squeeze_out"], outputs=["x_out"], axes=[1]
        )
        identity = onnx.helper.make_node(
            "Identity", inputs=["cond_in"], outputs=["cond_out"]
        )
        loop_body = onnx.helper.make_graph(
            [squeeze, unsqueeze, identity],
            "loop_body",
            [iter_count, cond_in, x_in],
            [cond_out, x_out],
        )
        self._test_op_upgrade(
            "Loop",
            12,
            [[], "", [2, 1]],
            [[2, 1]],
            [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT],
            attrs={"body": loop_body},
        )

    def test_LpNormalization(self) -> None:
        self._test_op_upgrade("LpNormalization", 1)

    def test_LpPool(self) -> None:
        # 1->2 adapter is missing
        self._test_op_upgrade(
            "LpPool", 2, [[1, 1, 5, 5]], [[1, 1, 4, 4]], attrs={"kernel_shape": [2, 2]}
        )

    def test_LRN_1(self) -> None:
        self._test_op_upgrade("LRN", 1, attrs={"size": 3})

    def test_LRN_2(self) -> None:
        self._test_op_upgrade(
            "LRN", 1, [[2, 3, 4, 5]], [[2, 3, 4, 5]], attrs={"size": 3}
        )

    def test_LSTM_1(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "LSTM",
            7,
            [[5, 3, 4], [1, 24, 4], [1, 24, 4]],
            [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]],
            attrs={"hidden_size": 6},
        )

    def test_LSTM_2(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "LSTM",
            7,
            [[5, 3, 4], [2, 24, 4], [2, 24, 4]],
            [[5, 2, 3, 6], [2, 3, 6], [2, 3, 6]],
            attrs={"hidden_size": 6, "direction": "bidirectional"},
        )

    def test_LSTM_3(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "LSTM",
            7,
            [
                [5, 3, 4],
                [1, 24, 4],
                [1, 24, 4],
                [1, 48],
                [5],
                [1, 5, 6],
                [1, 5, 6],
                [1, 18],
            ],
            [[5, 1, 3, 6], [1, 3, 6], [1, 3, 6]],
            [
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
            ],
            attrs={"hidden_size": 6},
        )

    def test_MatMul_1(self) -> None:
        self._test_op_upgrade("MatMul", 1, [[2, 3], [3, 4]], [[2, 4]])

    def test_MatMul_2(self) -> None:
        self._test_op_upgrade("MatMul", 1, [[5, 2, 3], [5, 3, 4]], [[5, 2, 4]])

    def test_MatMulInteger_1(self) -> None:
        self._test_op_upgrade(
            "MatMulInteger",
            10,
            [[2, 3], [3, 4]],
            [[2, 4]],
            [TensorProto.INT8, TensorProto.INT8],
            [TensorProto.INT32],
        )

    def test_MatMulInteger_2(self) -> None:
        self._test_op_upgrade(
            "MatMulInteger",
            10,
            [[2, 3], [3, 4], [], []],
            [[2, 4]],
            [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8],
            [TensorProto.INT32],
        )

    def test_MatMulInteger_3(self) -> None:
        self._test_op_upgrade(
            "MatMulInteger",
            10,
            [[2, 3], [3, 4], [2], [4]],
            [[2, 4]],
            [TensorProto.INT8, TensorProto.INT8, TensorProto.INT8, TensorProto.INT8],
            [TensorProto.INT32],
        )

    def test_Max(self) -> None:
        self._test_op_upgrade(
            "Max",
            1,
            [[2, 3, 4], [2, 3, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_MaxPool_1(self) -> None:
        self._test_op_upgrade(
            "MaxPool", 1, [[1, 1, 5, 5]], [[1, 1, 4, 4]], attrs={"kernel_shape": [2, 2]}
        )

    def test_MaxPool_2(self) -> None:
        self._test_op_upgrade(
            "MaxPool",
            8,
            [[1, 1, 5, 5]],
            [[1, 1, 4, 4], [1, 1, 4, 4]],
            output_types=[TensorProto.FLOAT, TensorProto.INT64],
            attrs={"kernel_shape": [2, 2]},
        )

    def test_MaxRoiPool(self) -> None:
        self._test_op_upgrade(
            "MaxRoiPool",
            1,
            [[2, 3, 20, 20], [4, 5]],
            [[4, 3, 3, 3]],
            attrs={"pooled_shape": [3, 3]},
        )

    def test_MaxUnpool(self) -> None:
        self._test_op_upgrade(
            "MaxUnpool",
            9,
            [[1, 1, 5, 5], [1, 1, 5, 5]],
            [[1, 1, 6, 6]],
            [TensorProto.FLOAT, TensorProto.INT64],
            attrs={"kernel_shape": [2, 2]},
        )

    def test_Mean(self) -> None:
        self._test_op_upgrade(
            "Mean",
            1,
            [[2, 3, 4], [2, 3, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_MeanVarianceNormalization(self) -> None:
        self._test_op_upgrade("MeanVarianceNormalization", 9, attrs={"axes": [1, 2]})

    def test_Min(self) -> None:
        self._test_op_upgrade(
            "Min",
            1,
            [[2, 3, 4], [2, 3, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_Mish(self) -> None:
        self._test_op_upgrade("Mish", 18)

    def test_Mod_1(self) -> None:
        self._test_op_upgrade("Mod", 10, [[2, 3], [2, 3]], [[2, 3]])

    def test_Mod_2(self) -> None:
        self._test_op_upgrade("Mod", 10, [[2, 3], [2, 3]], [[2, 3]], attrs={"fmod": 1})

    def test_Mul(self) -> None:
        self._test_op_upgrade(
            "Mul",
            1,
            [[2, 3, 4], [2, 1, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_Multinomial(self) -> None:
        self._test_op_upgrade(
            "Multinomial",
            7,
            [[3, 5]],
            [[3, 7]],
            output_types=[TensorProto.INT32],
            attrs={"sample_size": 7},
        )

    def test_Neg(self) -> None:
        self._test_op_upgrade("Neg", 1, attrs={"consumed_inputs": [0]})

    def test_NegativeLogLikelihoodLoss_1(self) -> None:
        self._test_op_upgrade(
            "NegativeLogLikelihoodLoss",
            12,
            [[3, 4, 5], [3, 5]],
            [[]],
            [TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_NegativeLogLikelihoodLoss_2(self) -> None:
        self._test_op_upgrade(
            "NegativeLogLikelihoodLoss",
            12,
            [[3, 4, 5], [3, 5], [4]],
            [[]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
        )

    def test_NonMaxSuppression(self) -> None:
        self._test_op_upgrade(
            "NonMaxSuppression",
            10,
            [[2, 3, 4], [3, 5, 6]],
            [[2, 3]],
            output_types=[TensorProto.INT64],
        )

    def test_NonZero(self) -> None:
        self._test_op_upgrade(
            "NonZero", 9, [[3, 3]], [[2, 4]], output_types=[TensorProto.INT64]
        )

    def test_Not(self) -> None:
        self._test_op_upgrade(
            "Not", 1, [[2, 3]], [[2, 3]], [TensorProto.BOOL], [TensorProto.BOOL]
        )

    def test_OneHot(self) -> None:
        self._test_op_upgrade("OneHot", 9, [[3, 4, 5], [], [2]], [[3, 4, 5, 6]])

    def test_Or(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "Or",
            7,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.BOOL, TensorProto.BOOL],
            [TensorProto.BOOL],
        )

    def test_Pad(self) -> None:
        # 1->2 adapter is missing
        self._test_op_upgrade(
            "Pad", 2, [[3, 4]], [[5, 8]], attrs={"pads": [1, 2, 1, 2], "value": 1.5}
        )

    def test_Pow(self) -> None:
        self._test_op_upgrade("Pow", 1, [[2, 3, 4], [2, 3, 4]], [[2, 3, 4]])

    def test_PRelu(self) -> None:
        self._test_op_upgrade(
            "PRelu",
            1,
            [[2, 3, 4], [2, 3, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_QLinearConv(self) -> None:
        self._test_op_upgrade(
            "QLinearConv",
            10,
            [[1, 3, 5, 5], [], [], [4, 3, 2, 2], [], [], [], []],
            [[1, 4, 4, 4]],
        )

    def test_QLinearMatMul(self) -> None:
        self._test_op_upgrade(
            "QLinearMatMul", 10, [[2, 3], [], [], [3, 4], [], [], [], []], [[2, 4]]
        )

    def test_QuantizeLinear(self) -> None:
        self._test_op_upgrade(
            "QuantizeLinear",
            10,
            [[3, 4, 5], [], []],
            [[3, 4, 5]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.UINT8],
            [TensorProto.UINT8],
        )

    def test_RandomNormal(self) -> None:
        self._test_op_upgrade(
            "RandomNormal", 1, [], [[3, 4, 5]], attrs={"shape": [3, 4, 5]}
        )

    def test_RandomNormalLike(self) -> None:
        like = helper.make_tensor(
            "a",
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(),
            raw=True,
        )
        self._test_op_upgrade(
            "RandomNormalLike", 1, [[3, 4, 5]], [[3, 4, 5]], initializer=[like]
        )

    def test_RandomUniform(self) -> None:
        self._test_op_upgrade(
            "RandomUniform", 1, [], [[3, 4, 5]], attrs={"shape": [3, 4, 5]}
        )

    def test_RandomUniformLike(self) -> None:
        like = helper.make_tensor(
            "a",
            TensorProto.FLOAT,
            dims=[3, 4, 5],
            vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(),
            raw=True,
        )
        self._test_op_upgrade(
            "RandomUniformLike", 1, [[3, 4, 5]], [[3, 4, 5]], initializer=[like]
        )

    def test_Range(self) -> None:
        start = helper.make_tensor("a", TensorProto.FLOAT, dims=[], vals=np.array([0]))
        end = helper.make_tensor("b", TensorProto.FLOAT, dims=[], vals=np.array([12]))
        step = helper.make_tensor("c", TensorProto.FLOAT, dims=[], vals=np.array([2]))
        self._test_op_upgrade(
            "Range", 11, [[], [], []], [[6]], initializer=[start, end, step]
        )

    def test_Reciprocal(self) -> None:
        self._test_op_upgrade("Reciprocal", 1, attrs={"consumed_inputs": [0]})

    def test_ReduceL1(self) -> None:
        self._test_op_upgrade("ReduceL1", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceL2(self) -> None:
        self._test_op_upgrade("ReduceL2", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceLogSum(self) -> None:
        self._test_op_upgrade("ReduceLogSum", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceLogSumExp(self) -> None:
        self._test_op_upgrade("ReduceLogSumExp", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceMean(self) -> None:
        self._test_op_upgrade("ReduceMean", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceMax(self) -> None:
        self._test_op_upgrade("ReduceMax", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceMin(self) -> None:
        self._test_op_upgrade("ReduceMin", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceProd(self) -> None:
        self._test_op_upgrade("ReduceProd", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceSum(self) -> None:
        self._test_op_upgrade("ReduceSum", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_ReduceSumSquare(self) -> None:
        self._test_op_upgrade("ReduceSumSquare", 1, [[3, 4, 5]], [[1, 1, 1]])

    def test_Relu(self) -> None:
        self._test_op_upgrade("Relu", 1, attrs={"consumed_inputs": [0]})

    def test_Reshape(self) -> None:
        self._test_op_upgrade(
            "Reshape",
            1,
            [[3, 4, 5]],
            [[3, 10, 2]],
            attrs={"consumed_inputs": [0], "shape": [3, 10, 2]},
        )

    def test_Resize(self) -> None:
        self._test_op_upgrade("Resize", 10, [[3, 4, 5], [3]], [[3, 8, 15]])

    def test_ReverseSequence(self) -> None:
        self._test_op_upgrade(
            "ReverseSequence",
            10,
            [[3, 4, 5], [4]],
            [[3, 4, 5]],
            [TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_RNN_1(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "RNN",
            7,
            [[5, 3, 4], [1, 6, 4], [1, 6, 4]],
            [[5, 1, 3, 6], [1, 3, 6]],
            attrs={"hidden_size": 6},
        )

    def test_RNN_2(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "RNN",
            7,
            [[5, 3, 4], [2, 6, 4], [2, 6, 4]],
            [[5, 2, 3, 6], [2, 3, 6]],
            attrs={"hidden_size": 6, "direction": "bidirectional"},
        )

    def test_RNN_3(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "RNN",
            7,
            [[5, 3, 4], [1, 6, 4], [1, 6, 4], [1, 12], [5], [1, 5, 6]],
            [[5, 1, 3, 6], [1, 3, 6]],
            [
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
            ],
            attrs={"hidden_size": 6},
        )

    def test_RoiAlign_1(self) -> None:
        self._test_op_upgrade(
            "RoiAlign",
            10,
            [[2, 3, 20, 20], [10, 4], [10]],
            [[10, 3, 1, 1]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_RoiAlign_2(self) -> None:
        self._test_op_upgrade(
            "RoiAlign",
            16,
            [[2, 3, 20, 20], [10, 4], [10]],
            [[10, 3, 1, 1]],
            [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.INT64],
            attrs={"coordinate_transformation_mode": "half_pixel"},
        )

    def test_Round(self) -> None:
        self._test_op_upgrade("Round", 11)

    def test_Scatter(self) -> None:
        self._test_op_upgrade(
            "Scatter",
            9,
            [[2, 3], [1, 2], [1, 2]],
            [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT],
        )

    def test_ScatterElements_1(self) -> None:
        self._test_op_upgrade(
            "ScatterElements",
            11,
            [[2, 3], [1, 2], [1, 2]],
            [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT],
        )

    def test_ScatterElements_2(self) -> None:
        self._test_op_upgrade(
            "ScatterElements",
            16,
            [[2, 3], [1, 2], [1, 2]],
            [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT],
            attrs={"reduction": "add"},
        )

    def test_ScatterND_1(self) -> None:
        self._test_op_upgrade(
            "ScatterND",
            11,
            [[2, 3], [1, 2], [1, 2]],
            [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT],
        )

    def test_ScatterND_2(self) -> None:
        self._test_op_upgrade(
            "ScatterND",
            16,
            [[2, 3], [1, 2], [1, 2]],
            [[2, 3]],
            [TensorProto.FLOAT, TensorProto.INT64, TensorProto.FLOAT],
            [TensorProto.FLOAT],
            attrs={"reduction": "mul"},
        )

    def test_Scan(self) -> None:
        sum_in = onnx.helper.make_tensor_value_info(
            "sum_in", onnx.TensorProto.FLOAT, [2]
        )
        next_in = onnx.helper.make_tensor_value_info(
            "next_in", onnx.TensorProto.FLOAT, [2]
        )
        sum_out = onnx.helper.make_tensor_value_info(
            "sum_out", onnx.TensorProto.FLOAT, [2]
        )
        scan_out = onnx.helper.make_tensor_value_info(
            "scan_out", onnx.TensorProto.FLOAT, [2]
        )
        add_node = onnx.helper.make_node(
            "Add", inputs=["sum_in", "next_in"], outputs=["sum_out"]
        )
        id_node = onnx.helper.make_node(
            "Identity", inputs=["sum_out"], outputs=["scan_out"]
        )
        body = onnx.helper.make_graph(
            [add_node, id_node], "scan_body", [sum_in, next_in], [sum_out, scan_out]
        )
        self._test_op_upgrade(
            "Scan",
            8,
            ["", [1, 2], [1, 3, 2]],
            [[1, 2], [1, 3, 2]],
            attrs={"body": body, "num_scan_inputs": 1},
        )

    def test_Selu(self) -> None:
        self._test_op_upgrade("Selu", 1, attrs={"consumed_inputs": [0]})

    def test_Shape(self) -> None:
        self._test_op_upgrade(
            "Shape", 1, [[3, 4, 5]], [[3]], output_types=[TensorProto.INT64]
        )

    def test_Shrink(self) -> None:
        self._test_op_upgrade("Shrink", 9)

    def test_Sigmoid(self) -> None:
        self._test_op_upgrade("Sigmoid", 1, attrs={"consumed_inputs": [0]})

    def test_Sign(self) -> None:
        self._test_op_upgrade("Sign", 9)

    def test_Sinh(self) -> None:
        self._test_op_upgrade("Sinh", 9)

    def test_Sin(self) -> None:
        self._test_op_upgrade("Sin", 7)

    def test_Size(self) -> None:
        self._test_op_upgrade(
            "Size", 1, [[3, 4, 5]], [[]], output_types=[TensorProto.INT64]
        )

    def test_Slice(self) -> None:
        self._test_op_upgrade(
            "Slice",
            1,
            [[3, 4, 5]],
            [[3, 2, 2]],
            attrs={"axes": [1, 2], "starts": [0, 1], "ends": [2, 3]},
        )

    def test_Softmax_0(self) -> None:
        self._test_op_upgrade("Softmax", 1, attrs={"axis": 0})

    def test_Softmax_1(self) -> None:
        self._test_op_upgrade("Softmax", 1, attrs={"axis": 1})

    def test_Softmax_2(self) -> None:
        self._test_op_upgrade("Softmax", 1, attrs={"axis": 2})

    def test_Softmax_3(self) -> None:
        self._test_op_upgrade("Softmax", 1, attrs={"axis": -1})

    def test_Softmax_4(self) -> None:
        self._test_op_upgrade("Softmax", 1, attrs={"axis": -2})

    def test_Softmax_5(self) -> None:
        self._test_op_upgrade("Softmax", 1, attrs={"axis": -3})

    def test_Softplus(self) -> None:
        self._test_op_upgrade("Softplus", 1)

    def test_Softsign(self) -> None:
        self._test_op_upgrade("Softsign", 1)

    def test_SoftmaxCrossEntropyLoss(self) -> None:
        self._test_op_upgrade(
            "SoftmaxCrossEntropyLoss",
            12,
            [[3, 4, 5, 6], [3, 6]],
            [[]],
            [TensorProto.FLOAT, TensorProto.INT64],
        )

    def test_SpaceToDepth(self) -> None:
        self._test_op_upgrade(
            "SpaceToDepth", 1, [[1, 3, 8, 8]], [[1, 12, 4, 4]], attrs={"blocksize": 2}
        )

    def test_Split(self) -> None:
        # 1->2 adapter is missing
        self._test_op_upgrade(
            "Split",
            2,
            [[3, 4, 7]],
            [[3, 4, 2], [3, 4, 1], [3, 4, 4]],
            attrs={"axis": 2, "split": [2, 1, 4]},
        )

    def test_Sqrt(self) -> None:
        self._test_op_upgrade("Sqrt", 1, attrs={"consumed_inputs": [0]})

    def test_Squeeze(self) -> None:
        self._test_op_upgrade("Squeeze", 1, [[2, 1, 3, 4, 1]], [[2, 3, 4]])

    def test_StringNormalizer(self) -> None:
        self._test_op_upgrade(
            "StringNormalizer",
            10,
            [[1, 3]],
            [[1, 3]],
            [TensorProto.STRING],
            [TensorProto.STRING],
            attrs={"case_change_action": "LOWER"},
        )

    def test_Sub(self) -> None:
        self._test_op_upgrade(
            "Sub",
            1,
            [[2, 3, 4], [2, 3, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_Sum(self) -> None:
        self._test_op_upgrade(
            "Sum",
            1,
            [[2, 3, 4], [2, 3, 4]],
            [[2, 3, 4]],
            attrs={"consumed_inputs": [0]},
        )

    def test_Tanh(self) -> None:
        self._test_op_upgrade("Tanh", 1, attrs={"consumed_inputs": [0]})

    def test_Tan(self) -> None:
        self._test_op_upgrade("Tan", 7)

    def test_TfIdfVectorizer(self) -> None:
        self._test_op_upgrade(
            "TfIdfVectorizer",
            9,
            [[3]],
            [[5]],
            attrs={
                "max_gram_length": 3,
                "max_skip_count": 1,
                "min_gram_length": 2,
                "mode": "TFIDF",
                "ngram_counts": [0, 20],
                "ngram_indexes": [3, 4],
            },
        )

    def test_ThresholdedRelu(self) -> None:
        self._test_op_upgrade("ThresholdedRelu", 10)

    def test_Tile(self) -> None:
        # 5->6 adapter is missing
        repeats = helper.make_tensor(
            "b", TensorProto.INT64, dims=[3], vals=np.array([1, 2, 3])
        )
        self._test_op_upgrade(
            "Tile",
            6,
            [[3, 4, 5], [3]],
            [[3, 8, 15]],
            [TensorProto.FLOAT, TensorProto.INT64],
            initializer=[repeats],
        )

    def test_TopK(self) -> None:
        self._test_op_upgrade(
            "TopK",
            1,
            [[3, 4, 5]],
            [[3, 4, 2], [3, 4, 2]],
            output_types=[TensorProto.FLOAT, TensorProto.INT64],
            attrs={"k": 2},
        )

    def test_Transpose(self) -> None:
        self._test_op_upgrade(
            "Transpose",
            1,
            [[1, 2, 5, 3, 7]],
            [[1, 7, 5, 2, 3]],
            attrs={"perm": [0, 4, 2, 1, 3]},
        )

    def test_Trilu(self) -> None:
        self._test_op_upgrade("Trilu", 14)

    def test_Unique_1(self) -> None:
        self._test_op_upgrade("Unique", 11, [[3, 4, 5]], [[None]])

    def test_Unique_2(self) -> None:
        self._test_op_upgrade(
            "Unique", 11, [[3, 4, 5]], [[3, None, 5]], attrs={"axis": 1}
        )

    def test_Unsqueeze(self) -> None:
        self._test_op_upgrade(
            "Unsqueeze", 1, [[3, 4, 5]], [[3, 4, 1, 5]], attrs={"axes": [2]}
        )

    def test_Upsample(self) -> None:
        self._test_op_upgrade(
            "Upsample",
            1,
            [[1, 3, 4, 5]],
            [[1, 3, 6, 10]],
            attrs={"width_scale": 2.0, "height_scale": 1.5},
        )

    def test_Where(self) -> None:
        self._test_op_upgrade(
            "Where",
            9,
            [[2, 3], [2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.BOOL, TensorProto.FLOAT, TensorProto.FLOAT],
        )

    def test_Xor(self) -> None:
        # 6->7 adapter is missing
        self._test_op_upgrade(
            "Xor",
            7,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.BOOL, TensorProto.BOOL],
            [TensorProto.BOOL],
        )

    def test_CastLike(self) -> None:
        self._test_op_upgrade(
            "CastLike",
            15,
            [[2, 3, 4], [2, 1, 4]],
            [[2, 3, 4]],
            input_types=[TensorProto.FLOAT, TensorProto.FLOAT16],
            output_types=[TensorProto.FLOAT16],
        )

    def test_LayerNormalization(self) -> None:
        self._test_op_upgrade(
            "LayerNormalization",
            17,
            [[2, 3, 4, 5], [4, 5], [4, 5]],
            [[2, 3, 4, 5]],
            input_types=[TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
            output_types=[TensorProto.FLOAT],
            attrs={"axis": 2},
        )

    def _test_window_function(self, window_function_name: str) -> None:
        size = helper.make_tensor("a", TensorProto.INT64, dims=[], vals=np.array([10]))
        self._test_op_upgrade(
            window_function_name,
            17,
            [[]],
            [[10]],
            [TensorProto.INT64],
            initializer=[size],
        )

    def test_BlackmanWindow(self) -> None:
        self._test_window_function("BlackmanWindow")

    def test_HannWindow(self) -> None:
        self._test_window_function("HannWindow")

    def test_HammingWindow(self) -> None:
        self._test_window_function("HammingWindow")

    def test_DFT(self) -> None:
        self._test_op_upgrade("DFT", 17, [[2, 16, 1], []], [[2, 16, 2]])
        self._test_op_upgrade("DFT", 17, [[2, 16, 2], []], [[2, 16, 2]])
        self._test_op_upgrade(
            "DFT", 17, [[2, 16, 1], []], [[2, 9, 2]], attrs={"onesided": 1}
        )
        self._test_op_upgrade(
            "DFT", 17, [[2, 16, 2], []], [[2, 9, 2]], attrs={"onesided": 1}
        )
        self._test_op_upgrade(
            "DFT", 17, [[2, 16, 1], []], [[2, 16, 2]], attrs={"inverse": 1}
        )
        self._test_op_upgrade(
            "DFT", 17, [[2, 16, 2], []], [[2, 16, 2]], attrs={"inverse": 1}
        )
        self._test_op_upgrade(
            "DFT", 17, [[2, 16, 2], []], [[2, 16, 2]], attrs={"inverse": 1, "axis": 0}
        )

    def _test_short_time_fourier_transform(self, operator_name: str) -> None:
        # Real
        signal = helper.make_tensor(
            "a",
            TensorProto.FLOAT,
            dims=[2, 64],
            vals=np.random.rand(2, 64).astype(np.float32),
        )
        frame_step = helper.make_tensor(
            "b", TensorProto.INT64, dims=[1], vals=np.array([8])
        )
        window = helper.make_tensor(
            "c", TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32)
        )
        self._test_op_upgrade(
            operator_name,
            17,
            [[2, 64], [1], [16]],
            [[2, 7, 16, 2]],
            [
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.INT64,
            ],
            initializer=[signal, frame_step, window],
        )

        # Real Onesided
        signal = helper.make_tensor(
            "a",
            TensorProto.FLOAT,
            dims=[2, 64],
            vals=np.random.rand(2, 64).astype(np.float32),
        )
        frame_step = helper.make_tensor(
            "b", TensorProto.INT64, dims=[1], vals=np.array([8])
        )
        window = helper.make_tensor(
            "c", TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32)
        )
        self._test_op_upgrade(
            operator_name,
            17,
            [[2, 64], [1], [16]],
            [[2, 7, 9, 2]],
            [
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.INT64,
            ],
            attrs={"onesided": 1},
            initializer=[signal, frame_step, window],
        )

        # Complex
        signal = helper.make_tensor(
            "a",
            TensorProto.FLOAT,
            dims=[2, 64, 2],
            vals=np.random.rand(2, 64, 2).astype(np.float32),
        )
        frame_step = helper.make_tensor(
            "b", TensorProto.INT64, dims=[1], vals=np.array([8])
        )
        window = helper.make_tensor(
            "c", TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32)
        )
        self._test_op_upgrade(
            operator_name,
            17,
            [[2, 64, 2], [1], [16]],
            [[2, 7, 16, 2]],
            [
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.INT64,
            ],
            initializer=[signal, frame_step, window],
        )

        # Complex Onesided
        signal = helper.make_tensor(
            "a",
            TensorProto.FLOAT,
            dims=[2, 64, 2],
            vals=np.random.rand(2, 64, 2).astype(np.float32),
        )
        frame_step = helper.make_tensor(
            "b", TensorProto.INT64, dims=[1], vals=np.array([8])
        )
        window = helper.make_tensor(
            "c", TensorProto.FLOAT, dims=[16], vals=np.ones(16).astype(np.float32)
        )
        frame_length = helper.make_tensor(
            "e", TensorProto.INT64, dims=[1], vals=np.array([16])
        )
        self._test_op_upgrade(
            operator_name,
            17,
            [[2, 64, 2], [1], [16]],
            [[2, 7, 9, 2]],
            [
                TensorProto.FLOAT,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.INT64,
            ],
            attrs={"onesided": 1},
            initializer=[signal, frame_step, window, frame_length],
        )

    def test_STFT(self) -> None:
        self._test_short_time_fourier_transform("STFT")

    def test_MelWeightMatrix(self) -> None:
        num_mel_bins = helper.make_tensor(
            "a", TensorProto.INT64, dims=[], vals=np.array([10])
        )
        dft_length = helper.make_tensor(
            "b", TensorProto.INT64, dims=[], vals=np.array([64])
        )
        sample_rate = helper.make_tensor(
            "c", TensorProto.INT64, dims=[], vals=np.array([0])
        )
        lower_edge_hertz = helper.make_tensor(
            "d", TensorProto.FLOAT, dims=[], vals=np.array([0])
        )
        upper_edge_hertz = helper.make_tensor(
            "e", TensorProto.FLOAT, dims=[], vals=np.array([1])
        )

        self._test_op_upgrade(
            "MelWeightMatrix",
            17,
            [[], [], [], [], []],
            [[33, 10]],
            [
                TensorProto.INT64,
                TensorProto.INT64,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
            ],
            initializer=[
                num_mel_bins,
                dft_length,
                sample_rate,
                lower_edge_hertz,
                upper_edge_hertz,
            ],
        )

        num_mel_bins = helper.make_tensor(
            "a", TensorProto.INT64, dims=[], vals=np.array([20])
        )
        dft_length = helper.make_tensor(
            "b", TensorProto.INT64, dims=[], vals=np.array([31])
        )
        sample_rate = helper.make_tensor(
            "c", TensorProto.INT64, dims=[], vals=np.array([0])
        )
        lower_edge_hertz = helper.make_tensor(
            "d", TensorProto.FLOAT, dims=[], vals=np.array([0])
        )
        upper_edge_hertz = helper.make_tensor(
            "e", TensorProto.FLOAT, dims=[], vals=np.array([1])
        )

        self._test_op_upgrade(
            "MelWeightMatrix",
            17,
            [[], [], [], [], []],
            [[16, 20]],
            [
                TensorProto.INT64,
                TensorProto.INT64,
                TensorProto.INT64,
                TensorProto.FLOAT,
                TensorProto.FLOAT,
            ],
            initializer=[
                num_mel_bins,
                dft_length,
                sample_rate,
                lower_edge_hertz,
                upper_edge_hertz,
            ],
        )

    def test_CenterCropPad(self) -> None:
        input_ = helper.make_tensor(
            "input",
            TensorProto.FLOAT,
            dims=[2, 4],
            vals=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        )
        shape = helper.make_tensor(
            "shape", TensorProto.INT64, dims=[2], vals=np.array([3, 3])
        )
        self._test_op_upgrade(
            "CenterCropPad",
            18,
            [[], []],
            [[3, 3]],
            [TensorProto.FLOAT, TensorProto.INT64],
            initializer=[input_, shape],
        )

    def test_BitwiseNot(self) -> None:
        self._test_op_upgrade(
            "BitwiseNot",
            18,
            [[2, 3]],
            [[2, 3]],
            [TensorProto.INT32],
            [TensorProto.INT32],
        )

    def test_BitwiseAnd(self) -> None:
        self._test_op_upgrade(
            "BitwiseAnd",
            18,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.INT16, TensorProto.INT16],
            [TensorProto.INT16],
        )

    def test_BitwiseOr(self) -> None:
        self._test_op_upgrade(
            "BitwiseOr",
            18,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.INT16, TensorProto.INT16],
            [TensorProto.INT16],
        )

    def test_BitwiseXor(self) -> None:
        self._test_op_upgrade(
            "BitwiseXor",
            18,
            [[2, 3], [2, 3]],
            [[2, 3]],
            [TensorProto.INT16, TensorProto.INT16],
            [TensorProto.INT16],
        )

    def test_GroupNormalization(self) -> None:
        self._test_op_upgrade(
            "GroupNormalization",
            18,
            [[3, 4, 2, 2], [1], [1]],
            [[3, 4, 2, 2]],
            attrs={"epsilon": 1e-5, "num_groups": 2},
        )

    def test_StringConcat(self) -> None:
        self._test_op_upgrade(
            "StringConcat",
            20,
            [[2, 3], [2, 3]],
            [[2, 3]],
        )

    def test_RegexFullMatch(self) -> None:
        self._test_op_upgrade(
            "RegexFullMatch",
            20,
            [[2, 3]],
            [[2, 3]],
            [TensorProto.STRING],
            [TensorProto.BOOL],
        )

    def test_ops_tested(self) -> None:
        # NOTE: This test is order dependent and needs to run last in this class
        all_schemas = onnx.defs.get_all_schemas()
        all_op_names = {schema.name for schema in all_schemas if schema.domain == ""}
        excluded_ops = {
            # Sequence-based and Optional-based ops disabled because
            # the version converter doesn't play nicely with sequences
            "ConcatFromSequence",
            "SequenceAt",
            "SequenceConstruct",
            "SequenceEmpty",
            "SequenceErase",
            "SequenceInsert",
            "SequenceLength",
            "SequenceMap",
            "SplitToSequence",
            "Optional",
            "OptionalGetElement",
            "OptionalHasElement",
            "StringSplit",
        }
        expected_tested_ops = all_op_names - excluded_ops

        untested_ops = expected_tested_ops - set(self.tested_ops)
        self.assertEqual(untested_ops, set())


if __name__ == "__main__":
    unittest.main()
