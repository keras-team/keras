# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np

import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


class TestTrainingTool(unittest.TestCase):
    def test_training_info_proto(self) -> None:
        # Inference graph.
        A_shape = [2, 2]
        A_name = "A"
        A = np.random.rand(*A_shape).astype(np.float32)
        A_initializer = numpy_helper.from_array(A, name=A_name)
        A_value_info = helper.make_tensor_value_info(A_name, TensorProto.FLOAT, A_shape)

        B_shape = [2, 2]
        B_name = "B"
        B = np.random.rand(*B_shape).astype(np.float32)
        B_initializer = numpy_helper.from_array(B, name=B_name)
        B_value_info = helper.make_tensor_value_info(B_name, TensorProto.FLOAT, B_shape)

        C_shape = [2, 2]
        C_name = "C"
        C_value_info = helper.make_tensor_value_info(C_name, TensorProto.FLOAT, C_shape)

        inference_node = helper.make_node(
            "MatMul", inputs=[A_name, B_name], outputs=[C_name]
        )

        inference_graph = helper.make_graph(
            [inference_node],
            "simple_inference",
            [A_value_info, B_value_info],
            [C_value_info],
            [A_initializer, B_initializer],
        )

        # Training graph
        X_shape = [2, 2]
        X_name = "X"
        X = np.random.rand(*X_shape).astype(np.float32)
        X_initializer = numpy_helper.from_array(X, name=X_name)
        X_value_info = helper.make_tensor_value_info(X_name, TensorProto.FLOAT, X_shape)

        Y_shape = [2, 2]
        Y_name = "Y"
        Y_value_info = helper.make_tensor_value_info(Y_name, TensorProto.FLOAT, Y_shape)

        node = helper.make_node(
            "MatMul",
            inputs=[X_name, C_name],  # tensor "C" is from inference graph.
            outputs=[Y_name],
        )

        training_graph = helper.make_graph(
            [node], "simple_training", [X_value_info], [Y_value_info], [X_initializer]
        )

        # Capture assignment of B <--- Y.
        training_info = helper.make_training_info(
            training_graph, [(B_name, Y_name)], None, None
        )

        # Create a model with both inference and training information.
        model = helper.make_model(inference_graph)
        # Check if the inference-only part is correct.
        onnx.checker.check_model(model)
        # Insert training information.
        new_training_info = model.training_info.add()
        new_training_info.CopyFrom(training_info)

        # Generate the actual training graph from training information so that
        # we can run onnx checker to check if the full training graph is a valid
        # graph. As defined in spec, full training graph forms by concatenating
        # corresponding fields.
        full_training_graph = helper.make_graph(
            list(model.graph.node) + list(model.training_info[0].algorithm.node),
            "full_training_graph",
            list(model.graph.input) + list(model.training_info[0].algorithm.input),
            list(model.graph.output) + list(model.training_info[0].algorithm.output),
            list(model.graph.initializer)
            + list(model.training_info[0].algorithm.initializer),
        )

        # Wrap full training graph as a ModelProto so that we can run checker.
        full_training_model = helper.make_model(full_training_graph)
        full_training_model_with_shapes = shape_inference.infer_shapes(
            full_training_model
        )
        onnx.checker.check_model(full_training_model_with_shapes)


if __name__ == "__main__":
    unittest.main()
