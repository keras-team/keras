# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import typing
import unittest

import onnx
import onnx.parser
import onnx.shape_inference


class TestModelInference(unittest.TestCase):
    def _check(self, model_text: str, *expected: int):
        """Check that the model inference infers the expected types for outputs.
        Restricted to the simple case of tensor types, so expected types specify
        only the element type (ints corresponding to onnx.TensorProto.DataType).
        """
        model = onnx.parser.parse_model(model_text)
        inferred = onnx.shape_inference.infer_shapes(model)
        outputs = inferred.graph.output
        for output, expected_elem_type in zip(outputs, expected):
            inferred_type = output.type
            self.assertTrue(inferred_type.HasField("tensor_type"))
            tensor_type = inferred_type.tensor_type
            self.assertTrue(tensor_type.HasField("elem_type"))
            elem_type = tensor_type.elem_type
            self.assertEqual(elem_type, expected_elem_type)

    def _check_inference_error(self, model_text: str):
        """Check that the model inference raises an InferenceError."""
        model = onnx.parser.parse_model(model_text)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.shape_inference.infer_shapes(model, True, True)

    def test_unknown_op(self):
        """Test that model inference handles unknown ops.
        This special treatment is to support custom ops.
        See comments in shape inference code for details.
        """
        model = """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (y)
            {
                y = SomeUnknownOp (x)
            }
        """
        # No output types are inferred for unknown ops.
        # But ensure that the inference does not fail.
        self._check(model)

    def test_mi_basic(self):
        """Test that model inference infers model output type."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17]
            >
            agraph (float[N] x) => (y)
            {
                y = Cast<to=6> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function(self):
        """Test use of functions."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x) => (y)
            {
                y = local.cast(x)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast (x) => (y)
            {
                y = Cast<to=6> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function_attr(self):
        """Test use of functions with attribute parameters."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x) => (y)
            {
                y = local.cast<target=6>(x)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast<target>(x) => (y)
            {
                y = Cast<to:int = @target> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function_subgraph_attr(self):
        """Test use of function attributes within subgraphs."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x, bool flag) => (y)
            {
                y = local.cast<target=6>(x, flag)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast<target>(x, flag) => (y)
            {
                y = If (flag) <
                    then_branch = g1 () => (z_then) { z_then = Cast<to:int = @target> (x) },
                    else_branch = g2 () => (z_else) { z_else = Cast<to:int = @target> (x) }
                    >
            }
        """
        self._check(model, onnx.TensorProto.INT32)

    def test_mi_function_multiple_calls(self):
        """Test use of multiple invocation of functions."""
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            agraph (float[N] x, bool flag) => (y, z)
            {
                y = local.cast<target=6>(x, flag)
                z = local.cast<target=7>(x, flag)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            cast<target>(x, flag) => (y)
            {
                y = If (flag) <
                    then_branch = g1 () => (z_then) { z_then = Cast<to:int = @target> (x) },
                    else_branch = g2 () => (z_else) { z_else = Cast<to:int = @target> (x) }
                    >
            }
        """
        self._check(model, onnx.TensorProto.INT32, onnx.TensorProto.INT64)

    def _check_shape(self, model_text: str, *expected: typing.Sequence[int]):
        """Check that the model inference infers the expected shapes for outputs.
        Restricted to the simple case of tensor type outputs with completely
        known shapes.
        """
        model = onnx.parser.parse_model(model_text)
        inferred = onnx.shape_inference.infer_shapes(model, True, True, True)
        outputs = inferred.graph.output
        for output, expected_shape in zip(outputs, expected):
            inferred_type = output.type
            self.assertTrue(inferred_type.HasField("tensor_type"))
            tensor_type = inferred_type.tensor_type
            self.assertTrue(tensor_type.HasField("shape"))
            inferred_shape = tensor_type.shape
            self.assertEqual(len(inferred_shape.dim), len(expected_shape))
            for inferred_dim, expected_dim in zip(inferred_shape.dim, expected_shape):
                self.assertTrue(inferred_dim.HasField("dim_value"))
                self.assertEqual(inferred_dim.dim_value, expected_dim)

    def test_mi_constant(self):
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17]
            >
            mymodel (float[4, 8, 16] x) => (y) {
                shape = Constant<value_ints=[8,4,16]>()
                y = Reshape(x, shape)
            }
            """
        self._check_shape(model, [8, 4, 16])

    def test_mi_constant_2(self):
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17]
            >
            mymodel (float[4, 8, 16] x) => (y) {
                shape = Constant<value_ints=[4,2,8]>()
                two = Constant<value_int=2>()
                shape2 = Mul(shape, two)
                y = Reshape(x, shape2)
            }
            """
        self._check_shape(model, [8, 4, 16])

    def test_mi_constant_in_function(self):
        model = """
            <
                ir_version: 7,
                opset_import: [ "" : 17, "local" : 1]
            >
            main (float x) => (y, z) {
                y, z = local.expand(x)
            }
            <
                opset_import: [ "" : 17 ],
                domain: "local"
            >
            expand (x) => (y, z) {
                shape1 = Constant<value = int64[2] {4,4}>()
                shape2 = Constant<value = int64[3] {8,8,8}>()
                z = Expand (x, shape2)
                y = Expand (x, shape1)
            }
            """
        self._check_shape(model, [4, 4], [8, 8, 8])

    def test_mi_function_default_attr(self):
        """Test use of default values of function attributes."""
        model = """
            <ir_version: 7, opset_import: [ "" : 17, "local" : 1]>
            agraph (float[N] x) => (y, z)
            {
                y = local.cast <target=6> (x) # casts to INT32 type (encoding value 6)
                z = local.cast (x)  # uses default-attribute value of 1 (FLOAT type)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            cast <target: int = 1> (x) => (y)
            {
                y = Cast <to:int = @target> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32, onnx.TensorProto.FLOAT)

    def test_mi_overloaded_function(self):
        """Test use of functions."""
        model = """
            <ir_version: 10, opset_import: [ "" : 17, "local" : 1]>
            agraph (float[N] x) => (y, z)
            {
                y = local.cast:to_int32 (x)
                z = local.cast:to_int64 (x)
            }
            <opset_import: [ "" : 17 ], domain: "local", overload: "to_int32">
            cast (x) => (y)
            {
                y = Cast<to=6> (x)
            }
            <opset_import: [ "" : 17 ], domain: "local", overload: "to_int64">
            cast (x) => (y)
            {
                y = Cast<to=7> (x)
            }
        """
        self._check(model, onnx.TensorProto.INT32, onnx.TensorProto.INT64)


if __name__ == "__main__":
    unittest.main()
