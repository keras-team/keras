# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
import unittest
from typing import Sequence

import numpy as np

import onnx.defs
import onnx.parser
from onnx import (
    GraphProto,
    SparseTensorProto,
    TensorProto,
    checker,
    helper,
    numpy_helper,
    shape_inference,
)


class TestChecker(unittest.TestCase):
    @property
    def _sample_float_tensor(self) -> TensorProto:
        np_array = np.random.randn(2, 3).astype(np.float32)
        return helper.make_tensor(
            name="test",
            data_type=TensorProto.FLOAT,
            dims=(2, 3),
            vals=np_array.reshape(6).tolist(),
        )

    def make_sparse(
        self,
        shape: Sequence[int],
        values: Sequence[int],
        indices_shape: Sequence[int],
        indices: Sequence[int],
        name: str = "spval",
    ) -> SparseTensorProto:
        sparse = SparseTensorProto()
        sparse.dims.extend(shape)
        nnz = len(values)

        sparse.values.CopyFrom(
            helper.make_tensor(name, TensorProto.INT64, (nnz,), values)
        )
        sparse.indices.CopyFrom(
            helper.make_tensor("spind", TensorProto.INT64, indices_shape, indices)
        )
        return sparse

    def test_check_node(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")

        checker.check_node(node)

    def test_check_node_input_marked_optional(self) -> None:
        # GivenTensorFill's input is marked optional, hence it is used in this test.
        node = helper.make_node("GivenTensorFill", [], ["Y"], name="test")
        checker.check_node(node)

        # Explicitly pass the empty string as optional
        node = helper.make_node("GivenTensorFill", [""], ["Y"], name="test")
        checker.check_node(node)

        # Input of RELU is not optional
        node = helper.make_node("Relu", [""], ["Y"], name="test")
        self.assertRaises(checker.ValidationError, checker.check_node, node)

    def test_check_function_nested(self) -> None:
        func_domain = "local"
        func_nested_opset_imports = [
            helper.make_opsetid("", 14),
            helper.make_opsetid(func_domain, 1),
        ]
        # nested identity/add function
        func_nested_identity_add_name = "func_nested_identity_add"
        func_nested_identity_add_inputs = ["a", "b"]
        func_nested_identity_add_outputs = ["c"]
        func_nested_identity_add_nodes = [
            helper.make_node("func_identity", ["a"], ["a1"], domain=func_domain),
            helper.make_node("func_identity", ["b"], ["b1"], domain=func_domain),
            helper.make_node("func_add", ["a1", "b1"], ["c"], domain=func_domain),
        ]
        func_nested_identity_add = helper.make_function(
            func_domain,
            func_nested_identity_add_name,
            func_nested_identity_add_inputs,
            func_nested_identity_add_outputs,
            func_nested_identity_add_nodes,
            func_nested_opset_imports,
        )
        checker.check_function(func_nested_identity_add)

    def test_check_graph_ir_version_3(self) -> None:
        ctx = checker.C.CheckerContext()
        ctx.ir_version = 3
        ctx.opset_imports = {"": onnx.defs.onnx_opset_version()}

        lex_ctx = checker.C.LexicalScopeContext()

        def check_ir_version_3(g: GraphProto) -> None:
            checker.check_graph(g, ctx, lex_ctx)

        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        check_ir_version_3(graph)

        graph.initializer.extend([self._sample_float_tensor])

        graph.initializer[0].name = "no-exist"

        self.assertRaises(checker.ValidationError, check_ir_version_3, graph)

        graph.initializer[0].name = "X"
        check_ir_version_3(graph)

    def test_check_graph(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        checker.check_graph(graph)

        graph.initializer.extend([self._sample_float_tensor])

        graph.initializer[0].name = "no-exist"
        checker.check_graph(graph)

        graph.initializer[0].name = "X"
        checker.check_graph(graph)

    def test_check_graph_types(self) -> None:
        # This is for https://github.com/onnx/onnx/issues/3849.
        # It confirms that type checking is performed
        # when checker.check_model is called with full_check=True

        node_div = helper.make_node("Div", ["X", "Y"], ["Z"], name="test_div")
        node_identity = helper.make_node("Identity", ["Z"], ["W"], name="test_identity")

        graph = helper.make_graph(
            [node_div, node_identity],
            "test",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
                # intentionally use a BOOL type which is not supported by the Div op.
                helper.make_tensor_value_info("Y", TensorProto.BOOL, [1, 2]),
            ],
            [helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 2])],
        )

        model = helper.make_model(graph, producer_name="test")

        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

        checker.check_graph(graph)

        graph = helper.make_graph(
            [node_div, node_identity],
            "test",
            [
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
                # intentionally use a Int32 type which is in conflict with Div's other input X.
                helper.make_tensor_value_info("Y", TensorProto.INT32, [1, 2]),
            ],
            [helper.make_tensor_value_info("W", TensorProto.FLOAT, [1, 2])],
        )

        model = helper.make_model(graph, producer_name="test")

        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

        checker.check_graph(graph)

    def test_check_graph_empty_initializer_name(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        checker.check_graph(graph)

        # Supply no name for the initializer
        graph.initializer.extend([self._sample_float_tensor])
        graph.initializer[0].name = ""
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_empty_sparse_initializer_name(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        checker.check_graph(graph)

        # Supply no name for the sparse_initializer
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81], "")
        graph.sparse_initializer.extend([sparse])
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_duplicate_init_names(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        checker.check_graph(graph)

        graph.initializer.extend([self._sample_float_tensor])
        graph.initializer[0].name = "X"

        # Add sparse initializer with the same name as above
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81], "X")
        graph.sparse_initializer.extend([sparse])
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_optional_input(self) -> None:
        # GivenTensorFill's input is marked optional, hence it is used in this test.
        node = helper.make_node("GivenTensorFill", [""], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        checker.check_graph(graph)

    def test_check_graph_ssa(self) -> None:
        relu1 = helper.make_node("Relu", ["X"], ["Z"], name="relu1")
        relu2 = helper.make_node("Relu", ["Y"], ["Z"], name="relu2")

        graph = helper.make_graph(
            [relu1, relu2],
            "test",
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2]),
            ],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_graph_topologically_sorted(self) -> None:
        n1 = helper.make_node("Scale", ["X"], ["Y"], scale=2.0, name="n1")
        n2 = helper.make_node("Scale", ["Y"], ["Z"], scale=3.0, name="n2")

        graph = helper.make_graph(
            [n2, n1],
            "test",
            inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )
        self.assertRaises(checker.ValidationError, checker.check_graph, graph)

    def test_check_model(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        model = helper.make_model(graph, producer_name="test")

        checker.check_model(model)

    def test_check_serialized_model(self) -> None:
        node = helper.make_node("Relu", ["X"], ["Y"], name="test")
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        model = helper.make_model(graph, producer_name="test")

        checker.check_model(model.SerializeToString())

    def test_check_old_model(self) -> None:
        node = helper.make_node("Pad", ["X"], ["Y"], paddings=(0, 0, 0, 0))
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        onnx_id = helper.make_opsetid("", 1)
        model = helper.make_model(graph, producer_name="test", opset_imports=[onnx_id])

        checker.check_model(model)

    def test_check_tensor(self) -> None:
        tensor = self._sample_float_tensor
        checker.check_tensor(tensor)

        tensor.raw_data = np.random.randn(2, 3).astype(np.float32).tobytes()
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_string_tensor(self) -> None:
        tensor = TensorProto()
        tensor.data_type = TensorProto.STRING
        tensor.dims.append(1)
        tensor.string_data.append(b"Test")
        checker.check_tensor(tensor)

        del tensor.string_data[:]
        tensor.raw_data = b"Test"
        # string data should not be stored in raw_data field
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_check_tensor_mismatched_field(self) -> None:
        tensor = self._sample_float_tensor
        tensor.data_type = TensorProto.INT32
        self.assertRaises(checker.ValidationError, checker.check_tensor, tensor)

    def test_nested_graph(self) -> None:
        n1 = helper.make_node("Scale", ["X"], ["Y"], scale=2.0, name="n1")
        n2 = helper.make_node("Scale", ["Y"], ["Z"], scale=3.0, name="n2")

        graph = helper.make_graph(
            [n1, n2],
            "nested",
            inputs=[],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )

        i1 = helper.make_node(
            "If", ["cond"], ["Z"], then_branch=graph, else_branch=graph
        )

        graph = helper.make_graph(
            [i1],
            "test",
            inputs=[
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
            ],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )

        checker.check_graph(graph)

    def test_nested_graph_without_subgraph_input_shape(self) -> None:
        n1 = helper.make_node("Scale", ["X"], ["Y"], scale=2.0, name="n1")
        n2 = helper.make_node("Scale", ["Y"], ["Z"], scale=3.0, name="n2")

        input_x = onnx.ValueInfoProto()
        input_x.name = "X"
        graph = helper.make_graph(
            [n1, n2],
            "nested",
            inputs=[],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )

        i1 = helper.make_node(
            "If", ["cond"], ["Z"], then_branch=graph, else_branch=graph
        )

        graph = helper.make_graph(
            [i1],
            "test",
            inputs=[
                helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
                helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2]),
            ],
            outputs=[helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 2])],
        )

        checker.check_graph(graph)

    @property
    def _sample_0_elem_tensor(self) -> TensorProto:
        np_array = np.random.randn(0, 3).astype(np.float32)
        return helper.make_tensor(
            name="test",
            data_type=TensorProto.FLOAT,
            dims=(0, 3),
            vals=np_array.reshape(0).tolist(),
        )

    def test_check_tensor_zero_elem(self) -> None:
        tensor = self._sample_0_elem_tensor
        checker.check_tensor(tensor)

    def test_check_removed_experimental_op(self) -> None:
        node = helper.make_node("ConstantFill", [], ["Y"], name="test", shape=[1, 2])
        checker.check_node(node)

    def test_skip_schema_check_on_non_standard_domain(self) -> None:
        node = helper.make_node(
            "NonExistOp", ["X"], ["Y"], name="test", domain="test.domain"
        )
        graph = helper.make_graph(
            [node],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])],
        )
        onnx_id = helper.make_opsetid("test.domain", 1)
        model = helper.make_model(graph, producer_name="test", opset_imports=[onnx_id])
        checker.check_model(model)

    def test_check_sparse_tensor(self) -> None:
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 81])
        checker.check_sparse_tensor(sparse)

    def test_check_sparse_tensor_invalid_index(self) -> None:
        # index value 181 is out-of-range
        sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 181])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_unordered(self) -> None:
        # index values are not in sorted order
        sparse = self.make_sparse([100], [13, 17, 19], [3], [27, 9, 81])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_coo_format(self) -> None:
        sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 2], [0, 9, 2, 7, 8, 1])
        checker.check_sparse_tensor(sparse)

    def test_check_sparse_tensor_coo_format_invalid_index(self) -> None:
        sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 2], [0, 9, 0, 27, 8, 1])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_coo_format_invalid_shape(self) -> None:
        sparse = self.make_sparse([10, 10], [13, 17, 19], [2, 3], [0, 9, 2, 7, 8, 1])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_tensor_coo_format_invalid_dim2(self) -> None:
        sparse = self.make_sparse([10, 10], [13, 17, 19], [3, 1], [0, 1, 2])
        self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)

    def test_check_sparse_matmul(self) -> None:
        M = 5
        N = 10
        # Create ValueInfoProto for input X of shape [N]
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N])
        # Create a [M,N] sparse-matrix constant C
        sparse_tensor = self.make_sparse([M, N], [2, 3, 1], [3], [3, 11, 37])
        node1 = helper.make_node("Constant", [], ["C"], sparse_value=sparse_tensor)
        # Create ValueInfoProto for output Y of shape [M]
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M])
        # Compute Y = C X
        node2 = helper.make_node("MatMul", ["C", "X"], ["Y"])
        # create graph
        graph = helper.make_graph([node1, node2], "sparse_matmul", [X], [Y])
        # check graph
        checker.check_graph(graph)

    def test_check_model_unsupported_input_type(self) -> None:
        N = 10
        X = helper.make_tensor_value_info("X", TensorProto.BOOL, [N])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [N])
        onnx_id = helper.make_opsetid("", 6)
        node = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph([node], "test_add_input", [X, Y], [Z])
        model = helper.make_model(graph, producer_name="test", opset_imports=[onnx_id])
        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

    def test_check_model_inconsistent_type(self) -> None:
        N = 10
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N])
        Y = helper.make_tensor_value_info("Y", TensorProto.INT32, [N])
        Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [N])
        onnx_id = helper.make_opsetid("", 6)
        node = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph([node], "test_add_input", [X, Y], [Z])
        model = helper.make_model(graph, producer_name="test", opset_imports=[onnx_id])
        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

    def test_check_model_unsupported_output_type(self) -> None:
        N = 10
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [N])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [N])
        Z = helper.make_tensor_value_info("Z", TensorProto.BOOL, [N])
        onnx_id = helper.make_opsetid("", 6)
        node = helper.make_node("Add", ["X", "Y"], ["Z"])
        graph = helper.make_graph([node], "test_add_input", [X, Y], [Z])
        model = helper.make_model(graph, producer_name="test", opset_imports=[onnx_id])
        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

    def test_loop_with_same_initializer_input_below_ir4(self) -> None:
        # This is for testing IR<4: tensors must exist both in initializer and input
        # shape_inference should allow different number of graph input and node input for Loop
        # Comes from a tf2onnx model

        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid("", 8)],
            ir_version=3,
            graph=helper.make_graph(
                name="test-loop",
                inputs=[
                    helper.make_tensor_value_info(
                        "input_0", TensorProto.INT32, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "while_maximum_iterations_0", TensorProto.INT64, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__18", TensorProto.INT64, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__17", TensorProto.FLOAT, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "Const_0", TensorProto.INT32, shape=[1]
                    ),
                ],
                outputs=[
                    helper.make_tensor_value_info(
                        "output_0", TensorProto.INT32, shape=[1]
                    )
                ],
                initializer=[
                    numpy_helper.from_array(
                        np.array(9223372036854775807, dtype=np.int64),
                        name="while_maximum_iterations_0",
                    ),
                    numpy_helper.from_array(
                        np.array([-1], dtype=np.int64), name="const_fold_opt__18"
                    ),
                    numpy_helper.from_array(
                        np.array(10.0, dtype=np.float32), name="const_fold_opt__17"
                    ),
                    numpy_helper.from_array(
                        np.array([1], dtype=np.int32), name="Const_0"
                    ),
                ],
                nodes=[
                    helper.make_node(
                        "Cast",
                        inputs=["input_0"],
                        outputs=["while_cond_158_while_Less__13_0"],
                        name="while_cond_158_while_Less__13",
                        domain="",
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Less",
                        inputs=[
                            "while_cond_158_while_Less__13_0",
                            "const_fold_opt__17",
                        ],
                        outputs=["while_cond_158_while_Less_0"],
                        name="while_cond_158_while_Less",
                        domain="",
                    ),
                    helper.make_node(
                        "Squeeze",
                        inputs=["while_cond_158_while_Less_0"],
                        outputs=["while_cond_158_while_Squeeze_0"],
                        name="while_cond_158_while_Squeeze",
                        domain="",
                    ),
                    helper.make_node(
                        "Loop",
                        inputs=[
                            "while_maximum_iterations_0",
                            "while_cond_158_while_Squeeze_0",
                            "input_0",
                            "Const_0",
                        ],
                        outputs=["while_loop_0", "while_loop_1"],
                        name="while_loop",
                        body=helper.make_graph(
                            name="while_body",
                            inputs=[
                                helper.make_tensor_value_info(
                                    "while_while_loop_counter_0",
                                    TensorProto.INT64,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "cond__15_0", TensorProto.BOOL, shape=[]
                                ),
                                helper.make_tensor_value_info(
                                    "while_placeholder_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "const_fold_opt__19", TensorProto.FLOAT, shape=[]
                                ),
                            ],
                            outputs=[
                                helper.make_tensor_value_info(
                                    "cond___while_Identity_graph_outputs_Identity__3_0",
                                    TensorProto.BOOL,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "while_Identity_2_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                            ],
                            initializer=[
                                numpy_helper.from_array(
                                    np.array(10.0, dtype=np.float32),
                                    name="const_fold_opt__19",
                                )
                            ],
                            nodes=[
                                helper.make_node(
                                    "Add",
                                    inputs=[
                                        "while_placeholder_0",
                                        "while_add_const_0_0",
                                    ],
                                    outputs=["while_Identity_2_0"],
                                    name="while_Add",
                                ),
                                helper.make_node(
                                    "Cast",
                                    inputs=["while_Identity_2_0"],
                                    outputs=["cond___while_Less__13_0"],
                                    name="cond___while_Less__13",
                                    domain="",
                                    to=TensorProto.FLOAT,
                                ),
                                helper.make_node(
                                    "Less",
                                    inputs=[
                                        "cond___while_Less__13_0",
                                        "const_fold_opt__19",
                                    ],
                                    outputs=["cond___while_Less_0"],
                                    name="cond___while_Less",
                                    domain="",
                                ),
                                helper.make_node(
                                    "Squeeze",
                                    inputs=["cond___while_Less_0"],
                                    outputs=[
                                        "cond___while_Identity_graph_outputs_Identity__3_0"
                                    ],
                                    name="cond___while_Squeeze",
                                    domain="",
                                ),
                            ],
                        ),
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=["while_loop_0"],
                        outputs=["Reshape_tensor_0"],
                        name="Reshape_tensor",
                        axes=[0],
                    ),
                    helper.make_node(
                        "Reshape",
                        inputs=["Reshape_tensor_0", "const_fold_opt__18"],
                        outputs=["output_0"],
                        name="Reshape",
                    ),
                ],
            ),
        )
        # Should not throw an error
        checker.check_model(model, full_check=True)

    def test_loop_with_different_initializer_input_below_ir4(self) -> None:
        # This is for testing IR<4: tensors must exist both in initializer and input
        # Testing an optional input which does not exist in initializers
        # Checker should throw an error said the missing input is not in initializers

        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid("", 8)],
            ir_version=3,
            graph=helper.make_graph(
                name="test-loop",
                inputs=[
                    helper.make_tensor_value_info(
                        "input_0", TensorProto.INT32, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "while_maximum_iterations_0", TensorProto.INT64, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__18", TensorProto.INT64, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__17", TensorProto.FLOAT, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "Const_0", TensorProto.INT32, shape=[1]
                    ),
                ],
                outputs=[
                    helper.make_tensor_value_info(
                        "output_0", TensorProto.INT32, shape=[1]
                    )
                ],
                initializer=[
                    numpy_helper.from_array(
                        np.array(9223372036854775807, dtype=np.int64),
                        name="while_maximum_iterations_0",
                    ),
                    numpy_helper.from_array(
                        np.array([-1], dtype=np.int64), name="const_fold_opt__18"
                    ),
                    numpy_helper.from_array(
                        np.array(10.0, dtype=np.float32), name="const_fold_opt__17"
                    ),
                    numpy_helper.from_array(
                        np.array([1], dtype=np.int32), name="Const_0"
                    ),
                ],
                nodes=[
                    helper.make_node(
                        "Cast",
                        inputs=["input_0"],
                        outputs=["while_cond_158_while_Less__13_0"],
                        name="while_cond_158_while_Less__13",
                        domain="",
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Less",
                        inputs=[
                            "while_cond_158_while_Less__13_0",
                            "const_fold_opt__17",
                        ],
                        outputs=["while_cond_158_while_Less_0"],
                        name="while_cond_158_while_Less",
                        domain="",
                    ),
                    helper.make_node(
                        "Squeeze",
                        inputs=["while_cond_158_while_Less_0"],
                        outputs=["while_cond_158_while_Squeeze_0"],
                        name="while_cond_158_while_Squeeze",
                        domain="",
                    ),
                    helper.make_node(
                        "Loop",
                        inputs=[
                            "while_maximum_iterations_0",
                            "while_cond_158_while_Squeeze_0",
                            "input_0",
                            "Const_0",
                        ],
                        outputs=["while_loop_0", "while_loop_1"],
                        name="while_loop",
                        body=helper.make_graph(
                            name="while_body",
                            inputs=[
                                helper.make_tensor_value_info(
                                    "while_while_loop_counter_0",
                                    TensorProto.INT64,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "cond__15_0", TensorProto.BOOL, shape=[]
                                ),
                                helper.make_tensor_value_info(
                                    "while_placeholder_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                                # The following input cannot be found in initializer and checker should throw an error
                                helper.make_tensor_value_info(
                                    "const_fold_opt__18", TensorProto.FLOAT, shape=[]
                                ),
                            ],
                            outputs=[
                                helper.make_tensor_value_info(
                                    "cond___while_Less__13_0",
                                    TensorProto.BOOL,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "while_Identity_2_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                            ],
                            initializer=[],
                            nodes=[
                                helper.make_node(
                                    "Add",
                                    inputs=[
                                        "while_placeholder_0",
                                        "while_add_const_0_0",
                                    ],
                                    outputs=["while_Identity_2_0"],
                                    name="while_Add",
                                ),
                                helper.make_node(
                                    "Cast",
                                    inputs=["while_Identity_2_0"],
                                    outputs=["cond___while_Less__13_0"],
                                    name="cond___while_Less__13",
                                    domain="",
                                    to=TensorProto.BOOL,
                                ),
                            ],
                        ),
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=["while_loop_0"],
                        outputs=["Reshape_tensor_0"],
                        name="Reshape_tensor",
                        axes=[0],
                    ),
                    helper.make_node(
                        "Reshape",
                        inputs=["Reshape_tensor_0", "const_fold_opt__18"],
                        outputs=["output_0"],
                        name="Reshape",
                    ),
                ],
            ),
        )
        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

    def test_loop_with_same_initializer_input_above_ir4(self) -> None:
        # This is for testing IR>=4:
        # Cannot use the same name as both a subgraph initializer and subgraph input

        model = helper.make_model(
            opset_imports=[helper.make_operatorsetid("", 11)],
            ir_version=6,
            graph=helper.make_graph(
                name="test-loop",
                inputs=[
                    helper.make_tensor_value_info(
                        "input_0", TensorProto.INT32, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "while_maximum_iterations_0", TensorProto.INT64, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__18", TensorProto.INT64, shape=[1]
                    ),
                    helper.make_tensor_value_info(
                        "const_fold_opt__17", TensorProto.FLOAT, shape=[]
                    ),
                    helper.make_tensor_value_info(
                        "Const_0", TensorProto.INT32, shape=[1]
                    ),
                ],
                outputs=[
                    helper.make_tensor_value_info(
                        "output_0", TensorProto.INT32, shape=[1]
                    )
                ],
                initializer=[
                    numpy_helper.from_array(
                        np.array(9223372036854775807, dtype=np.int64),
                        name="while_maximum_iterations_0",
                    ),
                    numpy_helper.from_array(
                        np.array([-1], dtype=np.int64), name="const_fold_opt__18"
                    ),
                    numpy_helper.from_array(
                        np.array(10.0, dtype=np.float32), name="const_fold_opt__17"
                    ),
                    numpy_helper.from_array(
                        np.array([1], dtype=np.int32), name="Const_0"
                    ),
                ],
                nodes=[
                    helper.make_node(
                        "Cast",
                        inputs=["input_0"],
                        outputs=["while_cond_158_while_Less__13_0"],
                        name="while_cond_158_while_Less__13",
                        domain="",
                        to=TensorProto.FLOAT,
                    ),
                    helper.make_node(
                        "Less",
                        inputs=[
                            "while_cond_158_while_Less__13_0",
                            "const_fold_opt__17",
                        ],
                        outputs=["while_cond_158_while_Less_0"],
                        name="while_cond_158_while_Less",
                        domain="",
                    ),
                    helper.make_node(
                        "Squeeze",
                        inputs=["while_cond_158_while_Less_0"],
                        outputs=["while_cond_158_while_Squeeze_0"],
                        name="while_cond_158_while_Squeeze",
                        domain="",
                    ),
                    helper.make_node(
                        "Loop",
                        inputs=[
                            "while_maximum_iterations_0",
                            "while_cond_158_while_Squeeze_0",
                            "input_0",
                            "Const_0",
                        ],
                        outputs=["while_loop_0", "while_loop_1"],
                        name="while_loop",
                        body=helper.make_graph(
                            name="while_body",
                            inputs=[
                                helper.make_tensor_value_info(
                                    "while_while_loop_counter_0",
                                    TensorProto.INT64,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "cond__15_0", TensorProto.BOOL, shape=[]
                                ),
                                helper.make_tensor_value_info(
                                    "while_placeholder_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                            ],
                            outputs=[
                                helper.make_tensor_value_info(
                                    "cond___while_Identity_graph_outputs_Identity__3_0",
                                    TensorProto.BOOL,
                                    shape=[],
                                ),
                                helper.make_tensor_value_info(
                                    "while_Identity_2_0", TensorProto.INT32, shape=[1]
                                ),
                                helper.make_tensor_value_info(
                                    "while_add_const_0_0", TensorProto.INT32, shape=[1]
                                ),
                            ],
                            # Cannot use the same name as both a subgraph initializer and subgraph input: while_while_loop_counter_0
                            initializer=[
                                numpy_helper.from_array(
                                    np.array(10, dtype=np.int64),
                                    name="while_while_loop_counter_0",
                                )
                            ],
                            nodes=[
                                helper.make_node(
                                    "Add",
                                    inputs=[
                                        "while_placeholder_0",
                                        "while_add_const_0_0",
                                    ],
                                    outputs=["while_Identity_2_0"],
                                    name="while_Add",
                                ),
                                helper.make_node(
                                    "Cast",
                                    inputs=["while_Identity_2_0"],
                                    outputs=["cond___while_Less__13_0"],
                                    name="cond___while_Less__13",
                                    domain="",
                                    to=TensorProto.FLOAT,
                                ),
                                helper.make_node(
                                    "Less",
                                    inputs=[
                                        "cond___while_Less__13_0",
                                        "while_while_loop_counter_0",
                                    ],
                                    outputs=["cond___while_Less_0"],
                                    name="cond___while_Less",
                                    domain="",
                                ),
                                helper.make_node(
                                    "Squeeze",
                                    inputs=["cond___while_Less_0"],
                                    outputs=[
                                        "cond___while_Identity_graph_outputs_Identity__3_0"
                                    ],
                                    name="cond___while_Squeeze",
                                    domain="",
                                ),
                            ],
                        ),
                    ),
                    helper.make_node(
                        "Unsqueeze",
                        inputs=["while_loop_0"],
                        outputs=["Reshape_tensor_0"],
                        name="Reshape_tensor",
                        axes=[0],
                    ),
                    helper.make_node(
                        "Reshape",
                        inputs=["Reshape_tensor_0", "const_fold_opt__18"],
                        outputs=["output_0"],
                        name="Reshape",
                    ),
                ],
            ),
        )
        self.assertRaises(
            shape_inference.InferenceError, checker.check_model, model, True
        )

    def test_empty_list_attribute(self):
        model = onnx.parser.parse_model(
            """
            <
                ir_version: 7,
                opset_import: [ "" : 17]
            >
            agraph (float[N] x) => (int64[M] y)
            {
                y = Constant <value_ints: ints = []>()
            }
        """
        )
        # Should not throw an error
        checker.check_model(model, full_check=True)
        model = onnx.parser.parse_model(
            """
            <
                ir_version: 7,
                opset_import: [ "" : 17]
            >
            agraph (float[N] x) => (float[M] y)
            {
                y = Constant <value_floats: floats = []>()
            }
        """
        )
        # Should not throw an error
        checker.check_model(model, full_check=True)

    def test_check_model_supports_unicode_path(self):
        input_tensor = helper.make_tensor_value_info(
            "input", onnx.TensorProto.FLOAT, [1]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", onnx.TensorProto.FLOAT, [1]
        )
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [input_tensor], [output_tensor])
        model = helper.make_model(graph, producer_name="test")

        with tempfile.TemporaryDirectory() as temp_dir:
            unicode_model_path = os.path.join(temp_dir, "模型モデル모델✨.onnx")
            onnx.save(model, unicode_model_path)
            checker.check_model(unicode_model_path, full_check=True)

    def test_graph_output_is_defined(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x) => (float[N] y, float[N] z)
            {
                y = Add(x, x)
            }
            # Error: z is not defined
        """
        )
        self.assertRaises(checker.ValidationError, checker.check_model, model)

    def test_graph_output_is_defined_within_sub_graph(self):
        model = onnx.parser.parse_model(
            """
            <ir_version: 7, opset_import: [ "" : 17]>
            agraph (float[N] x, bool cond) => (float[N] y)
            {
                sum = Add (x, x)
                prod = Mul (x, x)
                y = If (cond) <
                    then_branch = then_graph () => (sum) {},
                    else_branch = else_graph () => (prod) {}
                >
            }
            # Error: sum/prod are accessible inside if-then-else branches, but cannot
            # be used as outputs of the then/else branch implicitly.
            # An explicit "Identity(sum)" must be used to return sum as output.
        """
        )
        self.assertRaises(checker.ValidationError, checker.check_model, model)


if __name__ == "__main__":
    unittest.main()
