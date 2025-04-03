# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore
from __future__ import annotations

import unittest

import numpy as np

import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.reference as orf


def create_model():
    """The following model is equivalent to the following function.

    .. code-block:: python

        from onnx importonnx.TensorProto
        from onnx.helper import oh.make_tensor

        from onnxscript import script
        from onnxscript.onnx_opset import opset15 as op
        from onnxscript.onnx_types import FLOAT

        @script()
        def loop_range_cond_only(A: FLOAT["N"]) -> FLOAT["N"]:
            T = A
            cond = op.Constant(value=make_tensor("true",onnx.TensorProto.BOOL, [1], [1]))
            while cond:
                T = T + A
                cond = op.ReduceSum(T) > -10
            return T

        model = loop_range_cond_only.to_model_proto()
    """
    opset_imports = [
        oh.make_opsetid("", 15),
    ]
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    sparse_initializers = []
    functions = []
    inputs.append(oh.make_tensor_value_info("A", onnx.TensorProto.FLOAT, shape=("N",)))
    nodes.append(
        oh.make_node(
            "Constant",
            [],
            ["cond"],
            value=onh.from_array(np.array([True], dtype=np.bool_), name="value"),
        )
    )
    nodes.append(
        oh.make_node(
            "Constant",
            [],
            ["true"],
            value=onh.from_array(np.array(True, dtype=np.bool_), name="value"),
        )
    )

    def _make_local_graph_body():
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []
        inputs.append(
            oh.make_tensor_value_info("infinite_loop", onnx.TensorProto.INT64, shape=[])
        )
        inputs.append(
            oh.make_tensor_value_info("cond", onnx.TensorProto.BOOL, shape=[])
        )
        inputs.append(oh.make_tensor_value_info("T", onnx.TensorProto.UNDEFINED, []))
        nodes.append(oh.make_node("Add", ["T", "A"], ["T_0"]))
        nodes.append(oh.make_node("ReduceSum", ["T_0"], ["tmp"]))
        nodes.append(
            oh.make_node(
                "Constant",
                [],
                ["int64_m10"],
                value=onh.from_array(np.array(-10, dtype=np.int64), name="value"),
            )
        )
        nodes.append(oh.make_node("CastLike", ["int64_m10", "tmp"], ["int64_m10_cast"]))
        nodes.append(oh.make_node("Greater", ["tmp", "int64_m10_cast"], ["cond_1"]))
        nodes.append(oh.make_node("Identity", ["cond_1"], ["cond_out"]))
        outputs.append(
            oh.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, shape=[])
        )
        outputs.append(oh.make_tensor_value_info("T_0", onnx.TensorProto.UNDEFINED, []))
        graph = oh.make_graph(
            nodes,
            "loop_body",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        return graph

    body = _make_local_graph_body()
    nodes.append(oh.make_node("Loop", ["", "true", "A"], ["T_2"], body=body))
    outputs.append(
        oh.make_tensor_value_info("T_2", onnx.TensorProto.FLOAT, shape=("N",))
    )
    graph = oh.make_graph(
        nodes,
        "loop_range_cond_only",
        inputs,
        outputs,
        initializers,
        sparse_initializer=sparse_initializers,
    )
    model = oh.make_model(graph, functions=functions, opset_imports=opset_imports)
    return model


class TestReferenceEvaluatorModel(unittest.TestCase):
    def test_loop_fft(self):
        model = create_model()
        session = orf.ReferenceEvaluator(model)
        session.run(None, {"A": -np.arange(10).astype(np.float32)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
