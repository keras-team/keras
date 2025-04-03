# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest
from typing import Sequence

from shape_inference_test import TestShapeInferenceHelper

import onnx
import onnx.helper
import onnx.parser
import onnx.shape_inference
from onnx import AttributeProto, TypeProto

float_type_ = onnx.helper.make_tensor_type_proto(1, None)
uint8_type_ = onnx.helper.make_tensor_type_proto(2, None)
int8_type_ = onnx.helper.make_tensor_type_proto(3, None)
int32_type_ = onnx.helper.make_tensor_type_proto(6, None)
float16_type_ = onnx.helper.make_tensor_type_proto(10, None)
no_type_ = TypeProto()


class TestFunctionInference(TestShapeInferenceHelper):
    def _check(
        self,
        function_text: str,
        input_types: Sequence[TypeProto],
        attributes: Sequence[AttributeProto],
        expected_output_types: Sequence[TypeProto],
    ):
        function = onnx.parser.parse_function(function_text)
        result = onnx.shape_inference.infer_function_output_types(
            function, input_types, attributes
        )
        self.assertEqual(len(expected_output_types), len(result))
        for expected, actual in zip(expected_output_types, result):
            self._compare_value_infos(expected, actual)

    def _check_fails(
        self,
        function_text: str,
        input_types: Sequence[TypeProto],
        attributes: Sequence[AttributeProto],
    ):
        function = onnx.parser.parse_function(function_text)

        def invoke_inference():
            onnx.shape_inference.infer_function_output_types(
                function, input_types, attributes
            )

        self.assertRaises(onnx.shape_inference.InferenceError, invoke_inference)

    def test_fi_basic(self):
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            f (y, z) => (w) {
                x = Add(y, z)
                w = Mul(x, y)
            }
        """
        self._check(code, [float_type_, float_type_], [], [float_type_])
        self._check(code, [int32_type_, int32_type_], [], [int32_type_])
        self._check_fails(code, [float_type_, int32_type_], [])

    def test_fi_attribute(self):
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            CastTo <dtype> (x) => (y) {
                y = Cast <to : int = @dtype> (x)
            }
        """
        dtype_6 = onnx.helper.make_attribute("dtype", 6)
        self._check(code, [float_type_], [dtype_6], [int32_type_])

        dtype_10 = onnx.helper.make_attribute("dtype", 10)
        self._check(code, [float_type_], [dtype_10], [float16_type_])

    def test_fi_optional_input(self):
        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            DoReduce (x, axes) => (y) {
                y = ReduceMax (x, axes)
            }
        """
        # We can omit the type for a missing trailing optional parameter
        self._check(code, [float_type_], [], [float_type_])
        # Or, we can pass in a default-value of TypeProto() for a missing optional parameter
        self._check(code, [float_type_, no_type_], [], [float_type_])

        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            Quantize (x, scale, zero_point) => (y) {
                y = QuantizeLinear (x, scale, zero_point)
            }
        """
        # If the optional third parameter is specified, it determines the output type.
        self._check(code, [float_type_, float_type_, int8_type_], [], [int8_type_])
        self._check(code, [float_type_, float_type_, uint8_type_], [], [uint8_type_])
        # If the optional third parameter is omitted, the output type is uint8 (default).
        self._check(code, [float_type_, float_type_, no_type_], [], [uint8_type_])

        code = """
            <opset_import: [ "" : 18 ], domain: "local">
            DoClip (x, min, max) => (y) {
                y = Clip (x, min, max)
            }
        """
        # A test-case with a non-trailing missing optional parameter
        self._check(code, [float_type_, no_type_, float_type_], [], [float_type_])

        # A failing test-case with a non-trailing missing optional parameter
        self._check_fails(code, [float_type_, no_type_, int8_type_], [])


if __name__ == "__main__":
    unittest.main()
