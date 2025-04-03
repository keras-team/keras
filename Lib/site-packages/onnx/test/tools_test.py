# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import numpy as np
from numpy.testing import assert_allclose

import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.defs import onnx_opset_version
from onnx.reference import ReferenceEvaluator
from onnx.tools import update_model_dims
from onnx.tools.replace_constants import replace_initializer_by_constant_of_shape


class TestToolsFunctions(unittest.TestCase):
    def test_update_inputs_outputs_dim(self) -> None:
        node_def = helper.make_node(
            "Conv",
            inputs=["x", "W"],
            outputs=["y"],
            kernel_shape=[3, 3],
            strides=[2, 2],
        )
        graph_def = helper.make_graph(
            [node_def],
            "test",
            [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 5, 5]),
                helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 1, 3, 3]),
            ],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        )
        model_def = helper.make_model(graph_def, producer_name="test")
        updated_def = update_model_dims.update_inputs_outputs_dims(
            model_def,
            {
                "x": [1, 1, "x1", -1],
                "W": [1, 1, 3, 3],
            },
            {
                "y": [1, 1, -1, -1],
            },
        )
        onnx.checker.check_model(updated_def)
        self.assertEqual(
            updated_def.graph.input[0].type.tensor_type.shape.dim[2].dim_param, "x1"
        )
        self.assertEqual(
            updated_def.graph.input[0].type.tensor_type.shape.dim[3].dim_param, "x_3"
        )
        self.assertEqual(
            updated_def.graph.output[0].type.tensor_type.shape.dim[2].dim_param, "y_2"
        )
        self.assertEqual(
            updated_def.graph.output[0].type.tensor_type.shape.dim[3].dim_param, "y_3"
        )

    def test_replace_initializer(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = numpy_helper.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = numpy_helper.from_array(value, name="C")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node1 = helper.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = helper.make_node("Sub", ["AX", "C"], ["Y"])
        graph = helper.make_graph([node1, node2], "lr", [X], [Y], [A, C])
        model_def = helper.make_model(graph)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = {n.op_type for n in repl.graph.node}
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        assert_allclose(y1, y2)

    def test_replace_constant(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = numpy_helper.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = numpy_helper.from_array(value, name="C")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node0 = helper.make_node("Constant", [], ["A"], value=A)
        node1 = helper.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = helper.make_node("Sub", ["AX", "C"], ["Y"])
        graph = helper.make_graph([node0, node1, node2], "lr", [X], [Y], [C])
        model_def = helper.make_model(graph)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = {n.op_type for n in repl.graph.node}
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        assert_allclose(y1, y2)

    def test_replace_range(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = numpy_helper.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = numpy_helper.from_array(value, name="C")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        node0 = helper.make_node("Constant", [], ["A"], value=A)
        node1 = helper.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = helper.make_node("Sub", ["AX", "C"], ["Y"])
        graph = helper.make_graph([node0, node1, node2], "lr", [X], [Y], [C])
        model_def = helper.make_model(graph)

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def, use_range=True)
        node_types = {n.op_type for n in repl.graph.node}
        self.assertIn("Range", node_types)
        self.assertNotIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        assert_allclose(y1.shape, y2.shape)

    def test_replace_constant_function(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = numpy_helper.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = numpy_helper.from_array(value, name="C")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        nodeC = helper.make_node("Constant", [], ["C"], value=C)
        node0 = helper.make_node("Constant", [], ["A"], value=A)
        node1 = helper.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = helper.make_node("Sub", ["AX", "C"], ["Y"])
        opset_imports = [
            helper.make_opsetid("", onnx_opset_version()),
            helper.make_opsetid("custom", 1),
        ]
        fct = helper.make_function(
            "custom",
            "unittest",
            ["X"],
            ["Y"],
            [nodeC, node0, node1, node2],
            opset_imports,
        )

        node = helper.make_node("unittest", ["X"], ["Y"], domain="custom")
        graph = helper.make_graph([node], "lr", [X], [Y], [C])
        model_def = helper.make_model(
            graph, functions=[fct], opset_imports=opset_imports
        )

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def)
        node_types = {n.op_type for n in repl.functions[0].node}
        self.assertIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y1[:, :] = 3.5
        y1[0, :] = 0.5
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        assert_allclose(y1, y2)

    def test_replace_range_function(self):
        dtype = np.float32
        value = np.random.randn(2, 100).astype(dtype)
        A = numpy_helper.from_array(value, name="A")
        value = np.array([1], dtype=dtype)
        C = numpy_helper.from_array(value, name="C")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        nodeC = helper.make_node("Constant", [], ["C"], value=C)
        node0 = helper.make_node("Constant", [], ["A"], value=A)
        node1 = helper.make_node("MatMul", ["X", "A"], ["AX"])
        node2 = helper.make_node("Sub", ["AX", "C"], ["Y"])
        opset_imports = [
            helper.make_opsetid("", onnx_opset_version()),
            helper.make_opsetid("custom", 1),
        ]
        fct = helper.make_function(
            "custom",
            "unittest",
            ["X"],
            ["Y"],
            [nodeC, node0, node1, node2],
            opset_imports,
        )

        node = helper.make_node("unittest", ["X"], ["Y"], domain="custom")
        graph = helper.make_graph([node], "lr", [X], [Y], [C])
        model_def = helper.make_model(
            graph, functions=[fct], opset_imports=opset_imports
        )

        x = np.array([1, 2, 4, 5, 5, 4]).astype(np.float32).reshape((3, 2))
        oinf1 = ReferenceEvaluator(model_def)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(model_def, use_range=True)
        node_types = {n.op_type for n in repl.functions[0].node}
        self.assertIn("Range", node_types)
        self.assertNotIn("ConstantOfShape", node_types)
        oinf2 = ReferenceEvaluator(repl)
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        assert_allclose(y1.shape, y2.shape)

    def test_replace_constant_graph(self):
        value = np.array([0], dtype=np.float32)
        zero = numpy_helper.from_array(value, name="zero")

        X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])

        rsum = helper.make_node("ReduceSum", ["X"], ["rsum"])
        cond = helper.make_node("Greater", ["rsum", "zero"], ["cond"])

        then_out = helper.make_tensor_value_info(
            "then_out", onnx.TensorProto.FLOAT, None
        )
        then_cst = numpy_helper.from_array(np.array([1] * 129).astype(np.float32))

        then_const_node = helper.make_node(
            "Constant", inputs=[], outputs=["then_out"], value=then_cst, name="cst1"
        )
        then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

        else_out = helper.make_tensor_value_info(
            "else_out", onnx.TensorProto.FLOAT, None
        )
        else_cst = numpy_helper.from_array(np.array([-1] * 129).astype(np.float32))
        else_const_node = helper.make_node(
            "Constant", inputs=[], outputs=["else_out"], value=else_cst, name="cst2"
        )
        else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

        if_node = onnx.helper.make_node(
            "If", ["cond"], ["Y"], then_branch=then_body, else_branch=else_body
        )
        graph = helper.make_graph([rsum, cond, if_node], "if", [X], [Y], [zero])
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", onnx_opset_version())]
        )
        self.assertNotIn("ConstantOfShape", str(onnx_model))

        x = np.ones((3, 2), dtype=np.float32)
        oinf1 = ReferenceEvaluator(onnx_model)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(onnx_model)
        self.assertIn("ConstantOfShape", str(repl))
        oinf2 = ReferenceEvaluator(repl)
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        y1 = y1.copy()
        y1[:] = 0.5
        assert_allclose(y1, y2)

    def test_replace_range_graph(self):
        value = np.array([0], dtype=np.float32)
        zero = numpy_helper.from_array(value, name="zero")

        X = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
        Y = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])

        rsum = helper.make_node("ReduceSum", ["X"], ["rsum"])
        cond = helper.make_node("Greater", ["rsum", "zero"], ["cond"])

        then_out = helper.make_tensor_value_info(
            "then_out", onnx.TensorProto.FLOAT, None
        )
        then_cst = numpy_helper.from_array(np.array([1] * 129).astype(np.float32))

        then_const_node = helper.make_node(
            "Constant", inputs=[], outputs=["then_out"], value=then_cst, name="cst1"
        )
        then_body = helper.make_graph([then_const_node], "then_body", [], [then_out])

        else_out = helper.make_tensor_value_info(
            "else_out", onnx.TensorProto.FLOAT, None
        )
        else_cst = numpy_helper.from_array(np.array([-1] * 129).astype(np.float32))
        else_const_node = helper.make_node(
            "Constant", inputs=[], outputs=["else_out"], value=else_cst, name="cst2"
        )
        else_body = helper.make_graph([else_const_node], "else_body", [], [else_out])

        if_node = onnx.helper.make_node(
            "If", ["cond"], ["Y"], then_branch=then_body, else_branch=else_body
        )
        graph = helper.make_graph([rsum, cond, if_node], "if", [X], [Y], [zero])
        onnx_model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", onnx_opset_version())]
        )
        self.assertNotIn("ConstantOfShape", str(onnx_model))

        x = np.ones((3, 2), dtype=np.float32)
        oinf1 = ReferenceEvaluator(onnx_model)
        y1 = oinf1.run(None, {"X": x})[0]  # type: ignore[index]
        repl = replace_initializer_by_constant_of_shape(onnx_model, use_range=True)
        self.assertNotIn("ConstantOfShape", str(repl))
        self.assertIn("Range", str(repl))
        oinf2 = ReferenceEvaluator(repl)
        y2 = oinf2.run(None, {"X": x})[0]  # type: ignore[index]
        assert_allclose(y1.shape, y2.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
