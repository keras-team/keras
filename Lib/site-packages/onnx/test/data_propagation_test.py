# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

# TODO: remove the following ignore after mypy upgrade in ONNX
from shape_inference_test import TestShapeInferenceHelper

import onnx.parser
from onnx import TensorProto
from onnx.helper import make_node, make_tensor, make_tensor_value_info


class TestDataPropagation(TestShapeInferenceHelper):
    def test_expand_symbolic_input(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT32, (3, 1, 2)), ("y", TensorProto.INT32, (1, 4, 2))],
            [
                make_node("Shape", ["y"], ["shape"]),
                make_node("Expand", ["x", "shape"], ["z"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.INT32, (3, 4, 2)),
            ],
            data_prop=True,
        )

    def test_constantofshape_with_symbolic_shape(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5))],
            [
                make_node("Shape", ["x"], ["shape"]),
                make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["y"],
                    value=make_tensor("value", TensorProto.INT32, (1,), (2,)),
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, (3,)),
                make_tensor_value_info("y", TensorProto.INT32, (3, 4, 5)),
            ],
            data_prop=True,
        )  # type: ignore

    def test_model_data_propagation(self) -> None:
        """Infer the shape of z by propagating the value of xshape."""
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18]>
            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {
                xshape = Shape (x)
                z = Expand (y, xshape)
            }
        """
        )
        self._assert_inferred(
            model,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 8, 16)),
            ],
            data_prop=True,
        )

    def test_data_prop_via_function(self) -> None:
        """Test value-propagation through function calls.
        Underlying core example is same as previous test_model_data_propagation.
        """
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18, "local" : 1 ]>
            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {
                xshape = local.GetShape (x)
                z = Expand (y, xshape)
            }
            <domain: "local", opset_import: [ "" : 18 ]>
            GetShape (x) => (shapeval) {
                shapeval = Shape(x)
            }
        """
        )
        self._assert_inferred(
            model,
            [
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 8, 16)),
            ],
            data_prop=True,
        )

    def test_multiple_calls_to_function(self) -> None:
        """Test value-propagation handles multiple calls to same function correctly.
        Underlying core example is same as previous test_model_data_propagation.
        """
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 18, "local" : 1 ]>
            agraph (float[4, 1, 16] x, float[1, 8, 16] y) => () {
                yshape = local.GetShape (y)
                xshape = local.GetShape (x)
                z = Expand (y, xshape)
                w = Expand (y, yshape)
            }
            <domain: "local", opset_import: [ "" : 18 ]>
            GetShape (x) => (shapeval) {
                shapeval = Shape(x)
            }
        """
        )
        self._assert_inferred(
            model,
            [
                make_tensor_value_info("yshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("xshape", TensorProto.INT64, (3,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 8, 16)),
                make_tensor_value_info("w", TensorProto.FLOAT, (1, 8, 16)),
            ],
            data_prop=True,
        )


if __name__ == "__main__":
    unittest.main()
