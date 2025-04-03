# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx.shape_inference
from onnx import ModelProto, TensorProto, TensorShapeProto, ValueInfoProto, helper
from onnx.helper import make_model, make_tensor_value_info


class TestSymbolicShape(unittest.TestCase):
    def _assert_valueinfo_shape(
        self, onnx_model: ModelProto, value_infos: list[ValueInfoProto]
    ) -> None:
        """Assert onnx_model.value_info should be the same as expected value_infos
        Instead of exact symbol, use -1 to represent symbolic shape in expected value_infos
        """
        for expected_vi in value_infos:
            shape = self._get_shape_from_name(onnx_model, expected_vi.name)
            assert shape is not None, f"{onnx_model}"
            if expected_vi.type.HasField("tensor_type"):
                expected_shape = expected_vi.type.tensor_type.shape
            elif expected_vi.type.HasField("sparse_tensor_type"):
                expected_shape = expected_vi.type.sparse_tensor_type.shape
            assert len(shape.dim) == len(expected_shape.dim), f"{onnx_model}"
            for dim_i, dim in enumerate(shape.dim):
                expected_dim = expected_shape.dim[dim_i]
                # -1 means it's a symbolic shape
                if expected_dim.dim_value == -1:
                    # symbolic dimension must exist
                    assert dim.dim_param, f"{onnx_model}"
                else:
                    assert dim.dim_value == expected_dim.dim_value, f"{onnx_model}"

    def _count_unique_dim_param_number(self, onnx_model: ModelProto) -> int:
        """Return the total number of unique symbolic shape"""
        symbol_shape_set = set()
        inputs = list(onnx_model.graph.input)
        outputs = list(onnx_model.graph.output)
        valueinfos = list(onnx_model.graph.value_info)
        for v in inputs + outputs + valueinfos:
            for dim in v.type.tensor_type.shape.dim:
                if dim.dim_param:
                    symbol_shape_set.add(dim.dim_param)
        return len(symbol_shape_set)

    def _get_shape_from_name(
        self, onnx_model: ModelProto, name: str
    ) -> TensorShapeProto | None:
        """Get shape from tensor_type or sparse_tensor_type according to given name"""
        inputs = list(onnx_model.graph.input)
        outputs = list(onnx_model.graph.output)
        valueinfos = list(onnx_model.graph.value_info)
        for v in inputs + outputs + valueinfos:
            if v.name == name:
                if v.type.HasField("tensor_type"):
                    return v.type.tensor_type.shape  # type: ignore
                if v.type.HasField("sparse_tensor_type"):
                    return v.type.sparse_tensor_type.shape  # type: ignore
        return None

    def test_concat_enable_symbolic(self) -> None:
        concat = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["C"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat, cast],
            inputs=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, "A"]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, None])
            ],
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(
            inferred_model, [make_tensor_value_info("C", TensorProto.FLOAT, (2, -1))]
        )
        # the symbolic shape of C and output should be the same
        assert self._get_shape_from_name(
            inferred_model, "C"
        ) == self._get_shape_from_name(inferred_model, "output")

    def test_two_symbolic_concat(self) -> None:
        concat1 = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        concat2 = helper.make_node(
            "Concat", inputs=["C", "D"], outputs=["E"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["E"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat1, concat2, cast],
            inputs=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, "A"]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, "D"]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, None])
            ],
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(
            inferred_model,
            [
                make_tensor_value_info("C", TensorProto.FLOAT, (2, -1)),
                make_tensor_value_info("E", TensorProto.FLOAT, (2, -1)),
            ],
        )
        # the symbolic shape of E and output should be the same
        assert self._get_shape_from_name(
            inferred_model, "E"
        ) == self._get_shape_from_name(inferred_model, "output")

    def test_duplicate_symbolic_shape(self) -> None:
        concat1 = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        concat2 = helper.make_node(
            "Concat", inputs=["C", "D"], outputs=["E"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["E"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat1, concat2, cast],
            inputs=[
                helper.make_tensor_value_info("A", TensorProto.FLOAT, [2, "unk__0"]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [2, 3]),
                helper.make_tensor_value_info("D", TensorProto.FLOAT, [2, "unk__1"]),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, [2, "unk__0"]
                )
            ],
        )

        onnx_model = make_model(graph_def)
        original_count = self._count_unique_dim_param_number(onnx_model)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        inferred_count = self._count_unique_dim_param_number(inferred_model)
        # to prevent duplicate so the inferred count will be count + 2
        # new symbol 'unk__2' and 'unk__3' should be generated
        # original: {'unk_0', 'unk__1'}
        # inferred: {'unk_0', 'unk__1', 'unk__2', 'unk__3'}
        assert inferred_count == original_count + 2, f"{inferred_model}{onnx_model}"

    def test_unknown_shape(self) -> None:
        concat = helper.make_node(
            "Concat", inputs=["A", "B"], outputs=["C"], name="Concat", axis=1
        )
        cast = onnx.helper.make_node(
            "Cast", inputs=["C"], outputs=["output"], to=TensorProto.FLOAT
        )
        graph_def = helper.make_graph(
            name="test_graph",
            nodes=[concat, cast],
            inputs=[
                helper.make_tensor_value_info(
                    "A", TensorProto.FLOAT, [3, None]
                ),  # unknown shape
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, None]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, None])
            ],
        )

        onnx_model = make_model(graph_def)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model, strict_mode=True)
        self._assert_valueinfo_shape(
            inferred_model, [make_tensor_value_info("C", TensorProto.FLOAT, (3, -1))]
        )
        # the symbolic shape of C and output should be the same
        # ('unk__0', 'unk__1')
        assert self._get_shape_from_name(
            inferred_model, "C"
        ) == self._get_shape_from_name(inferred_model, "output")


if __name__ == "__main__":
    unittest.main()
