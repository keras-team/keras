# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import unittest
from typing import Any, Sequence

import numpy as np
import pytest
from parameterized import parameterized

import onnx.shape_inference
from onnx import (
    ONNX_ML,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    SparseTensorProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    checker,
    defs,
    helper,
    numpy_helper,
)
from onnx.defs import (
    AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    ONNX_DOMAIN,
    ONNX_ML_DOMAIN,
    OpSchema,
    SchemaError,
)
from onnx.helper import (
    make_empty_tensor_value_info,
    make_node,
    make_opsetid,
    make_tensor,
    make_tensor_sequence_value_info,
    make_tensor_value_info,
)
from onnx.parser import parse_graph


def get_available_versions(schema: OpSchema) -> set[int]:
    versions: set[int] = set()
    for version in range(schema.since_version, 0, -1):
        try:
            versions.add(
                defs.get_schema(schema.name, version, schema.domain).since_version
            )
        except SchemaError:  # noqa: PERF203
            break
    return versions


ALL_OP_VERSIONS: dict[str, tuple[str, frozenset[int]]] = {
    schema.name: (schema.domain, frozenset(get_available_versions(schema)))
    for schema in defs.get_all_schemas()
}


def all_versions_for(op_name: str) -> list[tuple[str, int]]:
    domain, versions_set = ALL_OP_VERSIONS[op_name]
    if not versions_set:
        raise ValueError(f"No versions available for operator {op_name}")
    versions = sorted(versions_set)
    return [
        (
            f"version{version}",
            version,
        )
        for version in versions
        # FIXME(#5289): Reshape errors in self._make_graph when version <= 5.
        # Issue reference: https://github.com/onnx/onnx/issues/5289.
        if version > 5 or domain != ONNX_DOMAIN
    ]


class TestShapeInferenceHelper(unittest.TestCase):
    def _make_graph(
        self,
        seed_values: Sequence[str | tuple[str, TensorProto.DataType, Any]],
        nodes: list[NodeProto],
        value_info: list[ValueInfoProto],
        initializer: Sequence[TensorProto] | None = None,
    ) -> GraphProto:
        if initializer is None:
            initializer = []
        names_in_initializer = {x.name for x in initializer}
        input_value_infos = []
        # If the starting values are not also initializers,
        # introduce the starting values as the output of reshape,
        # so that the sizes are guaranteed to be unknown
        for seed_value in seed_values:
            if isinstance(seed_value, tuple):
                seed_name, proto_type = seed_value[:2]
                seed_value_info = make_tensor_value_info(*seed_value)
            else:
                seed_name, proto_type = seed_value, TensorProto.UNDEFINED
                seed_value_info = make_empty_tensor_value_info(seed_value)
            if seed_name in names_in_initializer:
                input_value_infos.append(seed_value_info)
            else:
                value_info.append(seed_value_info)
                input_value_infos.append(
                    make_tensor_value_info("SEED_" + seed_name, proto_type, ())
                )
                input_value_infos.append(
                    make_tensor_value_info(
                        "UNKNOWN_SHAPE_" + seed_name, TensorProto.INT64, (None,)
                    )
                )
                nodes[:0] = [
                    make_node(
                        "Reshape",
                        ["SEED_" + seed_name, "UNKNOWN_SHAPE_" + seed_name],
                        [seed_name],
                    )
                ]
        return helper.make_graph(
            nodes,
            "test",
            input_value_infos,
            [],
            initializer=initializer,
            value_info=value_info,
        )

    def _inferred(
        self, graph_or_model: GraphProto | ModelProto, **kwargs: Any
    ) -> ModelProto:
        data_prop = kwargs.pop("data_prop", False)
        if isinstance(graph_or_model, GraphProto):
            kwargs["producer_name"] = "onnx-test"
            orig_model = helper.make_model(graph_or_model, **kwargs)
        else:
            orig_model = graph_or_model
        inferred_model = onnx.shape_inference.infer_shapes(
            orig_model, strict_mode=True, data_prop=data_prop
        )
        checker.check_model(inferred_model)
        return inferred_model

    def _assert_inferred(
        self,
        graph_or_model: GraphProto | ModelProto,
        vis: list[ValueInfoProto],
        **kwargs: Any,
    ) -> None:
        graph = (
            graph_or_model
            if isinstance(graph_or_model, GraphProto)
            else graph_or_model.graph
        )
        names_in_vis = {x.name for x in vis}
        vis = [x for x in graph.value_info if x.name not in names_in_vis] + vis
        inferred_model = self._inferred(graph_or_model, **kwargs)
        inferred_vis = list(inferred_model.graph.value_info)
        vis = sorted(vis, key=lambda x: x.name)  # type: ignore[no-any-return]
        inferred_vis = sorted(inferred_vis, key=lambda x: x.name)  # type: ignore
        assert len(vis) == len(inferred_vis)
        for v, inferred_v in zip(vis, inferred_vis):
            self._compare_value_infos(v.type, inferred_v.type)

    def _compare_value_infos(
        self, vi_type: TypeProto, inferred_vi_type: TypeProto
    ) -> None:
        if vi_type.HasField("tensor_type"):
            assert inferred_vi_type.HasField("tensor_type")
            assert vi_type.tensor_type.HasField("elem_type")
            assert inferred_vi_type.tensor_type.HasField("elem_type")
            assert (
                vi_type.tensor_type.elem_type == inferred_vi_type.tensor_type.elem_type
            )
            assert vi_type.tensor_type.HasField(
                "shape"
            ) == inferred_vi_type.tensor_type.HasField("shape")
            if vi_type.tensor_type.HasField("shape"):
                assert len(vi_type.tensor_type.shape.dim) == len(
                    inferred_vi_type.tensor_type.shape.dim
                )
                for dim_i, dim in enumerate(vi_type.tensor_type.shape.dim):
                    inferred_dim = inferred_vi_type.tensor_type.shape.dim[dim_i]
                    # if it is a symbolic shape, make sure the inferred symbol has generated (dim_param)
                    if dim.dim_param:
                        assert (
                            dim.dim_param == inferred_dim.dim_param
                        ), f"\n{vi_type}\n{inferred_vi_type}\n"
                    else:
                        assert (
                            dim.dim_value == inferred_dim.dim_value
                        ), f"\n{vi_type}\n{inferred_vi_type}\n"
        elif vi_type.HasField("sequence_type"):
            assert inferred_vi_type.HasField("sequence_type")
            vi = vi_type.sequence_type.elem_type
            inferred_vi = inferred_vi_type.sequence_type.elem_type
            self._compare_value_infos(vi, inferred_vi)
        elif vi_type.HasField("optional_type"):
            assert inferred_vi_type.HasField("optional_type")
            vi = vi_type.optional_type.elem_type
            inferred_vi = inferred_vi_type.optional_type.elem_type
            self._compare_value_infos(vi, inferred_vi)
        elif vi_type.HasField("map_type"):
            assert inferred_vi_type.HasField("map_type")
            assert vi_type.map_type.key_type == vi_type.map_type.key_type
            self._compare_value_infos(
                vi_type.map_type.value_type, inferred_vi_type.map_type.value_type
            )
        elif vi_type == onnx.TypeProto():
            assert inferred_vi_type == onnx.TypeProto()
        else:
            raise NotImplementedError(
                "Unrecognized value info type in _compare_value_infos: ", str(vi_type)
            )

    def skipIf(self, condition, reason):
        if condition:
            pytest.skip(reason)


class TestShapeInference(TestShapeInferenceHelper):
    def test_empty_graph(self) -> None:
        graph = self._make_graph(["y"], [], [])
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    def _identity_prop(self, op: str, **kwargs: Any) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (30, 4, 5))],
            [make_node(op, "x", "y", **kwargs)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (30, 4, 5))]
        )

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose(self, _, version) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose_preexisting(self, _, version) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose_scalar(self, _, version) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, ())],
            [make_node("Transpose", ["X"], ["Y"])],
            [],
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info("Y", TensorProto.FLOAT, ())],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose_partial(self, _, version) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.UNDEFINED, (3, "a", "b"))],
        )  # type: ignore
        self._assert_inferred(
            graph,
            [make_tensor_value_info("Y", TensorProto.FLOAT, (3, 2, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose_preexisting_incorrect_shape(self, *_) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 5, 5))],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose_preexisting_incorrect_type(self, *_) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 2])],
            [make_tensor_value_info("Y", TensorProto.STRING, (3, 2, 4))],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    @parameterized.expand(all_versions_for("Transpose"))
    def test_transpose_incorrect_repeated_perm(self, *_) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Transpose", ["X"], ["Y"], perm=[1, 0, 1])],
            [],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    def _make_matmul_test_all_dims_known(
        self, version, shape1: Sequence[int], shape2: Sequence[int]
    ) -> None:
        expected_out_shape = np.matmul(
            np.arange(np.prod(shape1)).reshape(shape1),
            np.arange(np.prod(shape2)).reshape(shape2),
        ).shape
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, shape1), ("y", TensorProto.FLOAT, shape2)],
            [make_node("MatMul", ["x", "y"], ["z"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, expected_out_shape)],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("MatMul"))
    def test_matmul_all_dims_known(self, _, version) -> None:
        self._make_matmul_test_all_dims_known(version, (2,), (2,))

        self._make_matmul_test_all_dims_known(version, (4, 2), (2, 4))
        self._make_matmul_test_all_dims_known(version, (5, 2), (2, 4))
        self._make_matmul_test_all_dims_known(version, (5, 2), (2, 1))
        self._make_matmul_test_all_dims_known(version, (1, 2), (2, 3))
        self._make_matmul_test_all_dims_known(version, (2,), (2, 3))
        self._make_matmul_test_all_dims_known(version, (4, 2), (2,))
        self._make_matmul_test_all_dims_known(version, (1, 4, 2), (3, 2, 3))
        self._make_matmul_test_all_dims_known(version, (3, 4, 2), (3, 2, 3))
        self._make_matmul_test_all_dims_known(version, (5, 1, 4, 2), (1, 3, 2, 3))
        self._make_matmul_test_all_dims_known(version, (4, 2), (3, 2, 3))

    def _make_matmul_test_allow_unknown(
        self, version, shape1: Any, shape2: Any, expected_out_shape: Any
    ) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, shape1), ("y", TensorProto.FLOAT, shape2)],
            [make_node("MatMul", ["x", "y"], ["z"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, expected_out_shape)],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("MatMul"))
    def test_matmul_allow_unknown(self, _, version) -> None:
        self._make_matmul_test_allow_unknown(version, (None,), (None,), ())
        self._make_matmul_test_allow_unknown(version, (3,), (None,), ())
        self._make_matmul_test_allow_unknown(version, (2,), (2, "a"), ("a",))
        self._make_matmul_test_allow_unknown(version, (4, 2), (2, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown(version, (4, None), (2, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown(version, (4, None), (None, "a"), (4, "a"))
        self._make_matmul_test_allow_unknown(
            version, (1, 4, 2), ("a", 2, 5), ("a", 4, 5)
        )
        self._make_matmul_test_allow_unknown(
            version, (1, 3, 4, 2), ("a", 2, 5), (1, 3, 4, 5)
        )
        self._make_matmul_test_allow_unknown(version, (3,), None, None)
        self._make_matmul_test_allow_unknown(version, None, None, None)

    @parameterized.expand(all_versions_for("Cast"))
    def test_cast(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Cast", ["x"], ["y"], to=TensorProto.UINT8)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, (2, 4, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Cast"))
    @unittest.skip(
        "Issue #5960"
    )  # FIXME(#5960) propagateElemTypeFromAttributeToOutput does not validate against output type constraints
    def test_cast_to_complex(self, _, version) -> None:  # noqa: ARG002
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Cast", ["x"], ["y"], to=TensorProto.COMPLEX128)],
            [],
        )

        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    @parameterized.expand(all_versions_for("CastLike"))
    def test_cast_like(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3)), ("t", TensorProto.FLOAT16, ("N",))],
            [make_node("CastLike", ["x", "t"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT16, (2, 4, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Col2Im"))
    def test_col2im(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (1, 5, 5)),
                ("output_shape", TensorProto.INT64, (2,)),
                ("kernel_shape", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Col2Im", ["input", "output_shape", "kernel_shape"], ["output"]
                )
            ],
            [],
            initializer=[
                make_tensor("output_shape", TensorProto.INT64, (2,), (5, 5)),
                make_tensor("kernel_shape", TensorProto.INT64, (2,), (1, 5)),
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("output", TensorProto.FLOAT, (1, 1, 5, 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Col2Im"))
    def test_col2im_strides(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (1, 9, 4)),
                ("output_shape", TensorProto.INT64, (2,)),
                ("kernel_shape", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Col2Im",
                    ["input", "output_shape", "kernel_shape"],
                    ["output"],
                    strides=[2, 2],
                )
            ],
            [],
            initializer=[
                make_tensor("output_shape", TensorProto.INT64, (2,), (5, 5)),
                make_tensor("kernel_shape", TensorProto.INT64, (2,), (3, 3)),
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("output", TensorProto.FLOAT, (1, 1, 5, 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Col2Im"))
    def test_col2im_pads(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (1, 5, 15)),
                ("output_shape", TensorProto.INT64, (2,)),
                ("kernel_shape", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Col2Im",
                    ["input", "output_shape", "kernel_shape"],
                    ["output"],
                    pads=[0, 1, 0, 1],
                )
            ],
            [],
            initializer=[
                make_tensor("output_shape", TensorProto.INT64, (2,), (5, 5)),
                make_tensor("kernel_shape", TensorProto.INT64, (2,), (1, 5)),
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("output", TensorProto.FLOAT, (1, 1, 5, 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Col2Im"))
    def test_col2im_dilations(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (1, 4, 5)),
                ("output_shape", TensorProto.INT64, (2,)),
                ("kernel_shape", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Col2Im",
                    ["input", "output_shape", "kernel_shape"],
                    ["output"],
                    dilations=[1, 5],
                )
            ],
            [],
            initializer=[
                make_tensor("output_shape", TensorProto.INT64, (2,), (6, 6)),
                make_tensor("kernel_shape", TensorProto.INT64, (2,), (2, 2)),
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("output", TensorProto.FLOAT, (1, 1, 6, 6))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Col2Im"))
    def test_col2im_5d(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (1, 10, 12)),
                ("output_shape", TensorProto.INT64, (3,)),
                ("kernel_shape", TensorProto.INT64, (3,)),
            ],
            [
                make_node(
                    "Col2Im", ["input", "output_shape", "kernel_shape"], ["output"]
                )
            ],
            [],
            initializer=[
                make_tensor("output_shape", TensorProto.INT64, (3,), (3, 4, 5)),
                make_tensor("kernel_shape", TensorProto.INT64, (3,), (1, 1, 5)),
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("output", TensorProto.FLOAT, (1, 2, 3, 4, 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Concat"))
    def test_concat(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3)), ("y", TensorProto.FLOAT, (7, 4, 3))],
            [make_node("Concat", ["x", "y"], ["z"], axis=0)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, (9, 4, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Concat"))
    def test_concat_missing_shape(self, *_) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (2, 4, 3)),
                "y",
                ("z", TensorProto.FLOAT, (None, None, None)),
            ],
            [make_node("Concat", ["x", "y", "z"], ["out"], axis=0)],
            [],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    @parameterized.expand(all_versions_for("Concat"))
    def test_concat_3d_axis_2(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 2, 2)), ("y", TensorProto.FLOAT, (2, 2, 2))],
            [make_node("Concat", ["x", "y"], ["z"], axis=2)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, (2, 2, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Concat"))
    def test_concat_param(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ("a", 2)), ("y", TensorProto.FLOAT, ("a", 3))],
            [make_node("Concat", ["x", "y"], ["z"], axis=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, ("a", 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Concat"))
    def test_concat_param_single_input(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ("a", 2))],
            [make_node("Concat", ["x"], ["z"], axis=0)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, ("a", 2))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_dynamic_shape_known_rank(self, _, version) -> None:
        self.skipIf(version < 14, "Rank inference is added from Version 14")
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (2, 4, 3)), ("shape", TensorProto.INT64, (2,))],
            [make_node("Reshape", ["x", "shape"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, (None, None))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_dynamic_shape_symbolic(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (2, 4, 3)), ("shape", TensorProto.INT64, ("M",))],
            [make_node("Reshape", ["x", "shape"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, None)],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_dynamic_unknown_shape(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (2, 4, 3)), ("shape", TensorProto.INT64, None)],
            [make_node("Reshape", ["x", "shape"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, None)],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_static_shape(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (2, 4, 3)), ("shape", TensorProto.INT64, (2,))],
            [make_node("Reshape", ["x", "shape"], ["y"])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (3, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, (3, 8))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_static_shape_inferred(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (2, 4, 3)), ("shape", TensorProto.INT64, (3,))],
            [make_node("Reshape", ["x", "shape"], ["y"])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (3,), (0, 3, -1))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, (2, 3, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_static_shape_zero(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (1, 1, 1)), ("shape", TensorProto.INT64, (3,))],
            [make_node("Reshape", ["x", "shape"], ["y"])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (3,), (0, 1, 1))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, (1, 1, 1))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_static_shape_allowzero(self, _, version) -> None:
        self.skipIf(version < 14, "allowzero is added from Version 14")
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (1, 0, 0)),
                ("shape", TensorProto.INT64, (3,)),
            ],
            [make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (3,), (0, 1, 1))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.UINT8, (0, 1, 1))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Reshape"))
    def test_reshape_static_shape_constant(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT8, (2, 4, 3))],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (2,), (3, 8)),
                ),
                make_node("Reshape", ["x", "shape"], ["y"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, (2,)),
                make_tensor_value_info("y", TensorProto.UINT8, (3, 8)),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Upsample"))
    def test_upsample(self, _, version) -> None:
        if version == 7:
            graph = self._make_graph(
                [("x", TensorProto.INT32, (2, 4, 3, 5))],
                [make_node("Upsample", ["x"], ["y"], scales=[1.0, 1.1, 1.3, 1.9])],
                [],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )
        else:
            graph = self._make_graph(
                [
                    ("x", TensorProto.INT32, (2, 4, 3, 5)),
                    ("scales", TensorProto.FLOAT, (4,)),
                ],
                [make_node("Upsample", ["x", "scales"], ["y"])],
                [],
                initializer=[
                    make_tensor("scales", TensorProto.FLOAT, (4,), (1.0, 1.1, 1.3, 1.9))
                ],
            )

            def call_inference():
                self._assert_inferred(
                    graph,
                    [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
                    opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
                )

            if version == 9:
                call_inference()
            else:
                # Upsample is deprecated since Version 10.
                with self.assertRaises(onnx.checker.ValidationError) as cm:
                    call_inference()
                exception = cm.exception
                assert "Upsample is deprecated" in str(exception)

    @parameterized.expand(all_versions_for("Upsample"))
    def test_upsample_raw_data(self, _, version) -> None:
        if version == 7:
            graph = self._make_graph(
                [("x", TensorProto.INT32, (1, 3, 4, 5))],
                [make_node("Upsample", ["x"], ["y"], scales=[2.0, 1.1, 2.3, 1.9])],
                [],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.INT32, (2, 3, 9, 9))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )
        else:
            graph = self._make_graph(
                [
                    ("x", TensorProto.INT32, (2, 4, 3, 5)),
                    ("scales", TensorProto.FLOAT, (4,)),
                ],
                [make_node("Upsample", ["x", "scales"], ["y"])],
                [],
                initializer=[
                    make_tensor(
                        "scales",
                        TensorProto.FLOAT,
                        (4,),
                        vals=np.array([1.0, 1.1, 1.3, 1.9], dtype="<f4").tobytes(),
                        raw=True,
                    )
                ],
            )  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose

            def call_inference():
                self._assert_inferred(
                    graph,
                    [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
                    opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
                )

            if version == 9:
                call_inference()
            else:
                # Upsample is deprecated since Version 10.
                with self.assertRaises(onnx.checker.ValidationError) as cm:
                    call_inference()
                exception = cm.exception
                assert "Upsample is deprecated" in str(exception)

    @parameterized.expand(all_versions_for("Expand"))
    def test_expand(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT32, (3, 1)), ("shape", TensorProto.INT64, (3,))],
            [make_node("Expand", ["x", "shape"], ["y"])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (3,), (2, 1, 6))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 3, 6))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Expand"))
    def test_expand_scalar_input(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT32, ()), ("shape", TensorProto.INT64, (2,))],
            [make_node("Expand", ["x", "shape"], ["y"])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (4, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (4, 8))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Expand"))
    def test_expand_raw_data(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT32, (3, 1)), ("shape", TensorProto.INT64, (2,))],
            [make_node("Expand", ["x", "shape"], ["y"])],
            [],
            initializer=[
                make_tensor(
                    "shape",
                    TensorProto.INT64,
                    (2,),
                    vals=np.array([3, 4], dtype="<i8").tobytes(),
                    raw=True,
                )
            ],
        )  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (3, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Expand"))
    def test_expand_dynamic_shape(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (1, 2, None)),
                ("shape", TensorProto.INT64, (3,)),
            ],
            [make_node("Expand", ["x", "shape"], ["y"])],
            [],
            initializer=[],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (None, 2, None))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Expand"))
    def test_expand_symbolic_shape(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (1, 2, None)),
                ("shape", TensorProto.INT64, ("unk__0",)),
            ],
            [make_node("Expand", ["x", "shape"], ["y"])],
            [],
            initializer=[],
        )
        # if giving a symbolic shape, Expand should not infer any shape or rank inference
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, None)],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size(self, _, version) -> None:
        if version == 10:
            graph = self._make_graph(
                [
                    ("x", TensorProto.INT32, (2, 4, 3, 5)),
                    ("scales", TensorProto.FLOAT, (4,)),
                ],
                [make_node("Resize", ["x", "scales"], ["y"])],
                [],
                initializer=[
                    make_tensor("scales", TensorProto.FLOAT, (4,), (1.0, 1.1, 1.3, 1.9))
                ],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )
        elif version == 11:
            graph = self._make_graph(
                [
                    ("x", TensorProto.INT32, (2, 4, 3, 5)),
                    ("roi", TensorProto.FLOAT, (8,)),
                    ("scales", TensorProto.FLOAT, (4,)),
                    ("sizes", TensorProto.INT64, (4,)),
                ],
                [make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"])],
                [],
                initializer=[
                    make_tensor("sizes", TensorProto.INT64, (4,), (3, 5, 6, 7))
                ],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.INT32, (3, 5, 6, 7))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )
        else:
            graph = self._make_graph(
                [
                    ("x", TensorProto.INT32, (2, 4, 3, 5)),
                    ("roi", TensorProto.FLOAT, (8,)),
                    ("sizes", TensorProto.INT64, (4,)),
                ],
                [make_node("Resize", ["x", "roi", "", "sizes"], ["y"])],
                [],
                initializer=[
                    make_tensor("sizes", TensorProto.INT64, (4,), (3, 5, 6, 7))
                ],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.INT32, (3, 5, 6, 7))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size_axes_2_3(self, _, version) -> None:
        self.skipIf(version < 18, "axes is from Version 18")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (2,)),
            ],
            [make_node("Resize", ["x", "roi", "", "sizes"], ["y"], axes=(2, 3))],
            [],
            initializer=[make_tensor("sizes", TensorProto.INT64, (2,), (6, 7))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 6, 7))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size_axes_3_2(self, _, version) -> None:
        self.skipIf(version < 18, "axes is from Version 18")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (2,)),
            ],
            [make_node("Resize", ["x", "roi", "", "sizes"], ["y"], axes=(3, 2))],
            [],
            initializer=[make_tensor("sizes", TensorProto.INT64, (2,), (6, 7))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 7, 6))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size_not_larger(self, _, version) -> None:
        self.skipIf(
            version < 18,
            "keep_aspect_ratio_policy is from Version 18",
        )
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (3, 5)),
                ("roi", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Resize",
                    ["x", "roi", "", "sizes"],
                    ["y"],
                    keep_aspect_ratio_policy="not_larger",
                )
            ],
            [],
            initializer=[make_tensor("sizes", TensorProto.INT64, (2,), (6, 6))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (4, 6))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size_axes_2_3_not_larger(self, _, version) -> None:
        self.skipIf(
            version < 18,
            "axes & keep_aspect_ratio_policy are from Version 18",
        )
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Resize",
                    ["x", "roi", "", "sizes"],
                    ["y"],
                    axes=(2, 3),
                    keep_aspect_ratio_policy="not_larger",
                )
            ],
            [],
            initializer=[make_tensor("sizes", TensorProto.INT64, (2,), (6, 6))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 4, 6))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size_not_smaller(self, _, version) -> None:
        self.skipIf(
            version < 18,
            "keep_aspect_ratio_policy is from Version 18",
        )
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (3, 5)),
                ("roi", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Resize",
                    ["x", "roi", "", "sizes"],
                    ["y"],
                    keep_aspect_ratio_policy="not_smaller",
                )
            ],
            [],
            initializer=[make_tensor("sizes", TensorProto.INT64, (2,), (6, 6))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (6, 10))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_size_axes_2_3_not_smaller(self, _, version) -> None:
        self.skipIf(
            version < 18,
            "axes & keep_aspect_ratio_policy are from Version 18",
        )
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (2,)),
            ],
            [
                make_node(
                    "Resize",
                    ["x", "roi", "", "sizes"],
                    ["y"],
                    axes=(2, 3),
                    keep_aspect_ratio_policy="not_smaller",
                )
            ],
            [],
            initializer=[make_tensor("sizes", TensorProto.INT64, (2,), (6, 6))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 6, 10))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_scale(self, _, version) -> None:
        self.skipIf(version < 11, "roi input is from Version 11")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (8,)),
                ("scales", TensorProto.FLOAT, (4,)),
            ],
            [make_node("Resize", ["x", "roi", "scales"], ["y"])],
            [],
            initializer=[
                make_tensor("scales", TensorProto.FLOAT, (4,), (1.0, 1.1, 1.3, 1.9))
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_scale_axes_2_3(self, _, version) -> None:
        self.skipIf(version < 18, "axes is from Version 18")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (8,)),
                ("scales", TensorProto.FLOAT, (2,)),
            ],
            [make_node("Resize", ["x", "roi", "scales"], ["y"], axes=(2, 3))],
            [],
            initializer=[make_tensor("scales", TensorProto.FLOAT, (2,), (1.3, 1.9))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_scale_axes_3_2(self, _, version) -> None:
        self.skipIf(version < 18, "axes is from Version 18")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (2, 4, 3, 5)),
                ("roi", TensorProto.FLOAT, (8,)),
                ("scales", TensorProto.FLOAT, (2,)),
            ],
            [make_node("Resize", ["x", "roi", "scales"], ["y"], axes=(3, 2))],
            [],
            initializer=[make_tensor("scales", TensorProto.FLOAT, (2,), (1.9, 1.3))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 4, 3, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_scale_raw_data(self, _, version) -> None:
        self.skipIf(version < 11, "roi input is from Version 11")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (1, 3, 4, 5)),
                ("roi", TensorProto.FLOAT, (8,)),
                ("scales", TensorProto.FLOAT, (4,)),
            ],
            [make_node("Resize", ["x", "roi", "scales"], ["y"])],
            [],
            initializer=[
                make_tensor(
                    "scales",
                    TensorProto.FLOAT,
                    (4,),
                    vals=np.array([2.0, 1.1, 2.3, 1.9], dtype="<f4").tobytes(),
                    raw=True,
                )
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 3, 9, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_scale_and_size_but_one_is_empty(self, _, version) -> None:
        self.skipIf(version < 11, "roi input is from Version 11")
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (1, 3, 4, 5)),
                ("roi", TensorProto.FLOAT, (8,)),
                ("scales", TensorProto.FLOAT, (4,)),
                ("sizes", TensorProto.INT64, (0,)),
            ],
            [make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"])],
            [],
            initializer=[
                make_tensor(
                    "scales",
                    TensorProto.FLOAT,
                    (4,),
                    vals=np.array([2.0, 1.1, 2.3, 1.9], dtype="<f4").tobytes(),
                    raw=True,
                ),
                make_tensor(
                    "sizes",
                    TensorProto.INT64,
                    (0,),
                    vals=np.array([], dtype="<i8").tobytes(),
                    raw=True,
                ),
            ],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 3, 9, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Resize"))
    def test_resize_opset11_scales_is_empty(self, _, version) -> None:
        self.skipIf(version != 11, "This test only works for Version 11")
        # "scales" input in Resize in opset11 is not optional. It must be an empty tensor
        # if sizes is needed. Shape inference for Resize shall handle this case.
        graph = self._make_graph(
            [
                ("x", TensorProto.INT32, (1, 3, 4, 5)),
                ("roi", TensorProto.FLOAT, (8,)),
                ("scales", TensorProto.FLOAT, (0,)),
                ("sizes", TensorProto.INT64, (4,)),
            ],
            [make_node("Resize", ["x", "roi", "scales", "sizes"], ["y"])],
            [],
            initializer=[
                make_tensor(
                    "sizes",
                    TensorProto.INT64,
                    (4,),
                    vals=np.array(
                        [2, 6, 8, 10], dtype="<i8"
                    ).tobytes(),  # double in all dimensions
                    raw=True,
                ),
            ],
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT32, (2, 6, 8, 10))],
            opset_imports=[helper.make_opsetid("", version)],
        )

    @parameterized.expand(all_versions_for("Shape"))
    def test_shape(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ["x"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (3,))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Shape"))
    def test_shape_start_1(self, _, version) -> None:
        self.skipIf(version < 15, "start and end are from Version 15")
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ["x"], ["y"], start=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (2,))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Shape"))
    def test_shape_end_1(self, _, version) -> None:
        self.skipIf(version < 15, "start and end are from Version 15")
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ["x"], ["y"], end=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (1,))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Shape"))
    def test_shape_negative_start(self, _, version) -> None:
        self.skipIf(version < 15, "start and end are from Version 15")
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ["x"], ["y"], start=-1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (1,))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Shape"))
    def test_shape_clip1(self, _, version) -> None:
        self.skipIf(version < 15, "start and end are from Version 15")
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ["x"], ["y"], start=-5)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (3,))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Shape"))
    def test_shape_clip2(self, _, version) -> None:
        self.skipIf(version < 15, "start and end are from Version 15")
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))],
            [make_node("Shape", ["x"], ["y"], end=10)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (3,))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Size"))
    def test_size(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4, 3))], [make_node("Size", ["x"], ["y"])], []
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, ())],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Gather"))
    def test_gather(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 3)), ("i", TensorProto.INT64, (2,))],
            [make_node("Gather", ["x", "i"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (2, 3))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Gather"))
    def test_gather_axis1(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 3, 5)), ("i", TensorProto.INT64, (1, 2))],
            [make_node("Gather", ["x", "i"], ["y"], axis=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (4, 1, 2, 5))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Gather"))
    def test_gather_into_scalar(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,)), ("i", TensorProto.INT64, ())],
            [make_node("Gather", ["x", "i"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, ())],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("GatherElements"))
    def test_gather_elements(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 2)), ("i", TensorProto.INT64, (2, 2))],
            [make_node("GatherElements", ["x", "i"], ["y"], axis=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (2, 2))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("GatherElements"))
    def test_gather_elements_axis0(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("i", TensorProto.INT64, (2, 3))],
            [make_node("GatherElements", ["x", "i"], ["y"], axis=0)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (2, 3))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("Scatter"))
    def test_scatter(self, _, version) -> None:
        if version >= 11:
            # Scatter is deprecated in domain_version of 11.
            with self.assertRaises(onnx.checker.ValidationError) as cm:
                self._test_scatter(version)
            exception = cm.exception
            assert "Scatter is deprecated" in str(exception)
        else:
            self._test_scatter(version)

    def _test_scatter(self, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 3)),
                ("i", TensorProto.INT64, (2, 3)),
                ("u", TensorProto.FLOAT, (2, 3)),
            ],
            [make_node("Scatter", ["x", "i", "u"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (3, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )  # type: ignore

    @parameterized.expand(all_versions_for("Scatter"))
    def test_scatter_axis1(self, _, version) -> None:
        if version >= 11:
            # Scatter is deprecated in domain_version of 11.
            with self.assertRaises(onnx.checker.ValidationError) as cm:
                self._test_scatter_axis1(version)
            exception = cm.exception
            assert "Scatter is deprecated" in str(exception)
        else:
            self._test_scatter_axis1(version)

    def _test_scatter_axis1(self, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (1, 5)),
                ("i", TensorProto.INT64, (1, 2)),
                ("u", TensorProto.FLOAT, (1, 2)),
            ],
            [make_node("Scatter", ["x", "i", "u"], ["y"], axis=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (1, 5))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )  # type: ignore

    @parameterized.expand(all_versions_for("ScatterElements"))
    def test_scatter_elements(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 3)),
                ("i", TensorProto.INT64, (2, 3)),
                ("u", TensorProto.FLOAT, (2, 3)),
            ],
            [make_node("ScatterElements", ["x", "i", "u"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (3, 3))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("ScatterElements"))
    def test_scatter_elements_axis1(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (1, 5)),
                ("i", TensorProto.INT64, (1, 2)),
                ("u", TensorProto.FLOAT, (1, 2)),
            ],
            [make_node("ScatterElements", ["x", "i", "u"], ["y"], axis=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (1, 5))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("ScatterND"))
    def test_scatternd(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (4, 5, 6)),
                ("indices", TensorProto.INT64, (3, 3, 2)),
                ("updates", TensorProto.FLOAT, (3, 3, 6)),
            ],
            [make_node("ScatterND", ["x", "indices", "updates"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (4, 5, 6))],  # type: ignore
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("ScatterND"))
    def test_scatternd_noshape(self, _, version) -> None:
        # The shape of 'x_reshaped' cannot be inferred, since it is the output of a dynamic reshape.
        # Thus the shape of 'y' is also None.
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (4, 5, 6)),
                ("indices", TensorProto.INT64, (3, 3, 2)),
                ("updates", TensorProto.FLOAT, (3, 3, 6)),
                ("shape", TensorProto.INT64, ("M",)),
            ],
            [
                make_node("Reshape", ["x", "shape"], ["x_reshaped"]),
                make_node("ScatterND", ["x_reshaped", "indices", "updates"], ["y"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("x_reshaped", TensorProto.FLOAT, None),
                make_tensor_value_info("y", TensorProto.FLOAT, None),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )  # type: ignore

    @parameterized.expand(all_versions_for("Squeeze"))
    def test_squeeze(self, _, version) -> None:
        if version == 11:
            graph = self._make_graph(
                [("x", TensorProto.FLOAT, (1, 3, 1, 1, 2, 1))],
                [make_node("Squeeze", "x", "y", axes=[0, 2, 3, 5])],
                [],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.FLOAT, (3, 2))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )
        else:
            graph = self._make_graph(
                [
                    ("x", TensorProto.FLOAT, (1, 3, 1, 1, 2, 1)),
                    ("axes", TensorProto.INT64, (4,)),
                ],
                [make_node("Squeeze", ["x", "axes"], "y")],
                [],
                initializer=[
                    make_tensor("axes", TensorProto.INT64, (4,), (0, 2, 3, 5))
                ],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("y", TensorProto.FLOAT, (3, 2))],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
            )

    @parameterized.expand(all_versions_for("StringConcat"))
    def test_stringconcat(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, (2, 3, 4)),
                ("y", TensorProto.STRING, (2, 3, 4)),
            ],
            [make_node("StringConcat", ["x", "y"], "z")],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.STRING, (2, 3, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("StringConcat"))
    def test_stringconcat_broadcasting(self, _, version) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, (2, 3, 4)),
                ("y", TensorProto.STRING, (1, 3, 1)),
            ],
            [make_node("StringConcat", ["x", "y"], "z")],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.STRING, (2, 3, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("RegexFullMatch"))
    def test_regex_full_match(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.STRING, (2, 4, 3, 9))],
            [make_node("RegexFullMatch", ["x"], ["y"], pattern=r"^[A-Z][a-z]*$")],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.BOOL, (2, 4, 3, 9))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("RegexFullMatch"))
    def test_regex_full_match_empty_shape(self, _, version) -> None:
        graph = self._make_graph(
            [("x", TensorProto.STRING, ())],
            [make_node("RegexFullMatch", ["x"], ["y"], pattern=r"^[A-Z][a-z]*$")],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.BOOL, ())],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    def test_squeeze_no_axes_opset11(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (1, 3, 1, 1, 2, 1)),
            ],
            [make_node("Squeeze", ["x"], "y")],
            [],
        )
        operatorsetid = OperatorSetIdProto()
        operatorsetid.domain = ""
        operatorsetid.version = 11
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 2))]
        )

    def test_unsqueeze_regular(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 2)), ("axes", TensorProto.INT64, (4,))],
            [make_node("Unsqueeze", ["x", "axes"], "y")],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (4,), (0, 1, 3, 5))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 1, 3, 1, 2, 1))]
        )

    def test_unsqueeze_unsorted_axes(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("axes", TensorProto.INT64, (2,))],
            [make_node("Unsqueeze", ["x", "axes"], "y")],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (4, 0))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 3, 4, 5, 1))]
        )

    def test_unsqueeze_negative_axes(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("axes", TensorProto.INT64, (2,))],
            [make_node("Unsqueeze", ["x", "axes"], "y")],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (0, -1))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 3, 4, 5, 1))]
        )

    def test_unsqueeze_scalar(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ()), ("axes", TensorProto.INT64, ())],
            [make_node("Unsqueeze", ["x", "axes"], "y")],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (), (-1,))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1,))]
        )

    def test_slice_without_input_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2, "a")),
                ("starts", TensorProto.INT64, (1,)),
                ("ends", TensorProto.INT64, (1,)),
            ],
            [make_node("Slice", ["x", "starts", "ends"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (None, None, None))]
        )

    def test_slice_with_input_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends"], ["y"])],
            [],
            initializer=[
                make_tensor(
                    "starts",
                    TensorProto.INT64,
                    (2,),
                    vals=np.array([1, 0], dtype="<i8").tobytes(),
                    raw=True,
                ),  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
                make_tensor("ends", TensorProto.INT64, (2,), (2, 2)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 2))]
        )

    def test_slice_with_input_shape_containing_dim_params(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (1, "a", 1)),
                ("starts", TensorProto.INT64, (3,)),
                ("ends", TensorProto.INT64, (3,)),
            ],
            [make_node("Slice", ["x", "starts", "ends"], ["y"])],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (3,), (0, 0, 0)),
                make_tensor("ends", TensorProto.INT64, (3,), (1, 1, 1)),
            ],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, None, 1))])  # type: ignore

    def test_slice_with_input_shape_steps(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (5, 6, 7)),
                ("starts", TensorProto.INT64, (3,)),
                ("ends", TensorProto.INT64, (3,)),
                ("axes", TensorProto.INT64, (None)),
                ("steps", TensorProto.INT64, (3,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (3,), (1, 0, 0)),
                make_tensor("ends", TensorProto.INT64, (3,), (2, 6, 6)),
                make_tensor("steps", TensorProto.INT64, (3,), (1, 4, 3)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 2, 2))]
        )

    def test_slice_with_input_shape_axes(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 6, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
                ("steps", TensorProto.INT64, (None)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                make_tensor("ends", TensorProto.INT64, (2,), (2, 2)),
                make_tensor("axes", TensorProto.INT64, (2,), (0, 2)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 6, 2))]
        )

    def test_slice_unsorted_axes(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes"], "y")],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                make_tensor("ends", TensorProto.INT64, (2,), (2, 2)),
                make_tensor("axes", TensorProto.INT64, (2,), (1, 0)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 1))]
        )  # can handle unsorted axes

    def test_slice_giant_number(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes"], "y")],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                make_tensor("ends", TensorProto.INT64, (2,), (200, 22000)),
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 2))]
        )

    def test_slice_giant_step(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
                ("steps", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes", "steps"], "y")],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                make_tensor("ends", TensorProto.INT64, (2,), (200, 200)),
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
                make_tensor("steps", TensorProto.INT64, (2,), (1, 200)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 1))]
        )

    def test_slice_negative_end(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes"], "y")],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (2,), (1, 0)),
                make_tensor(
                    "ends", TensorProto.INT64, (2,), (200, -1)
                ),  # negative end means begin from end of a dimension (here end = 2 - 1 = 1)
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
            ],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 1))])  # type: ignore

    def test_slice_negative_start(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 2)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes"], "y")],
            [],
            initializer=[
                make_tensor(
                    "starts", TensorProto.INT64, (2,), (1, -2)
                ),  # negative start means begin from end of a dimension (here end = 2 - 2 = 0)
                make_tensor("ends", TensorProto.INT64, (2,), (200, 3)),
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
            ],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 2))])  # type: ignore

    def test_slice_negative_step(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4)),
                ("starts", TensorProto.INT64, (2,)),
                ("ends", TensorProto.INT64, (2,)),
                ("axes", TensorProto.INT64, (2,)),
                ("steps", TensorProto.INT64, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes", "steps"], "y")],
            [],
            initializer=[
                make_tensor(
                    "starts", TensorProto.INT64, (2,), (1, 4)
                ),  # 4 will be clamped to 3 since we are negative stepping
                make_tensor("ends", TensorProto.INT64, (2,), (200, 0)),
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
                make_tensor("steps", TensorProto.INT64, (2,), (1, -1)),
            ],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 3))])  # type: ignore

    def test_slice_variable_copy(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("a", 2)),
                ("starts", TensorProto.INT64, (1,)),
                ("ends", TensorProto.INT64, (1,)),
                ("axes", TensorProto.INT64, (1,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes"], "y")],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT64, (1,), (1,)),
                make_tensor("ends", TensorProto.INT64, (1,), (200,)),
                make_tensor("axes", TensorProto.INT64, (1,), (1,)),
            ],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, ("a", 1))])  # type: ignore

    def test_slice_variable_input_types(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.DOUBLE, (3, 2)),
                ("starts", TensorProto.INT32, (2,)),
                ("ends", TensorProto.INT32, (2,)),
                ("axes", TensorProto.INT32, (2,)),
            ],
            [make_node("Slice", ["x", "starts", "ends", "axes"], "y")],
            [],
            initializer=[
                make_tensor("starts", TensorProto.INT32, (2,), (1, 0)),
                make_tensor("ends", TensorProto.INT32, (2,), (200, 22000)),
                make_tensor("axes", TensorProto.INT32, (2,), (0, 1)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.DOUBLE, (2, 2))]
        )

    def test_conv(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5, 6, 7)),
                ("y", TensorProto.FLOAT, (5, 4, 2, 4, 3)),
            ],
            [
                make_node(
                    "Conv",
                    ["x", "y"],
                    "z",
                    pads=[0, 1, 1, 0, 0, 1],
                    dilations=[1, 2, 2],
                    strides=[1, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 5, 4, 1, 3))]
        )

    def test_conv_1d_simple(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 5)),
                ("y", TensorProto.FLOAT, (50, 4, 2)),
            ],
            [make_node("Conv", ["x", "y"], "z", dilations=[1])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 4))]
        )

    def test_conv_dilations(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 8, 8, 8)),
                ("y", TensorProto.FLOAT, (50, 4, 3, 3, 3)),
            ],
            [make_node("Conv", ["x", "y"], "z", dilations=[1, 2, 3])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 6, 4, 2))]
        )

    def test_conv_strides(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 8, 8, 8)),
                ("y", TensorProto.FLOAT, (50, 4, 3, 3, 3)),
            ],
            [make_node("Conv", ["x", "y"], "z", strides=[1, 2, 3])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 6, 3, 2))]
        )

    def test_conv_pads(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 7, 6, 4)),
                ("y", TensorProto.FLOAT, (50, 4, 3, 3, 3)),
            ],
            [make_node("Conv", ["x", "y"], "z", pads=[1, 1, 2, 0, 1, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 6, 6, 6))]
        )

    def test_conv_auto_pad(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 7, 6, 4)),
                ("y", TensorProto.FLOAT, (50, 4, 4, 3, 2)),
            ],
            [make_node("Conv", ["x", "y"], "z", auto_pad="SAME_UPPER")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 7, 6, 4))]
        )

    def test_conv_auto_pads(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 7, 6, 4)),
                ("y", TensorProto.FLOAT, (50, 4, 4, 3, 2)),
            ],
            [
                make_node(
                    "Conv", ["x", "y"], "z", auto_pad="SAME_UPPER", strides=[2, 2, 1]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 4, 3, 4))]
        )

    def test_conv_auto_pad_dilation(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 65, 64, 63)),
                ("y", TensorProto.FLOAT, (50, 4, 4, 3, 2)),
            ],
            [
                make_node(
                    "Conv", ["x", "y"], "z", auto_pad="SAME_UPPER", dilations=[2, 3, 4]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 65, 64, 63))],
        )

    def test_conv_group(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 8, 8, 8)),
                ("y", TensorProto.FLOAT, (4, 1, 8, 8, 8)),
            ],
            [make_node("Conv", ["x", "y"], "z", group=4)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 4, 1, 1, 1))]
        )

    def test_conv_only_one_pos(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 5)),
                ("y", TensorProto.FLOAT, (50, 4, 5)),
            ],
            [make_node("Conv", ["x", "y"], "z", strides=[2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, 1))]
        )

    def test_conv_partial_missing_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, None, 6, 4)),
                ("y", TensorProto.FLOAT, (50, 4, 3, 3, 3)),
            ],
            [make_node("Conv", ["x", "y"], "z", pads=[1, 1, 2, 0, 1, 2])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 50, None, 6, 6))])  # type: ignore

    def test_conv_partial_missing_weight_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 7, 6, 4)),
                ("y", TensorProto.FLOAT, (50, 4, None, 3, 3)),
            ],
            [make_node("Conv", ["x", "y"], "z", pads=[1, 1, 2, 0, 1, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, None)]
        )

    def test_average_pool_auto_pads(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (30, 4, 7, 6, 4))],
            [
                make_node(
                    "AveragePool",
                    ["x"],
                    "z",
                    auto_pad="SAME_UPPER",
                    kernel_shape=[4, 3, 2],
                    strides=[2, 2, 1],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 4, 4, 3, 4))]
        )

    def test_average_pool_with_dilations(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], dilations=[2, 2]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_average_pool_with_same_upper_padding_and_stride_and_dilation(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "AveragePool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_UPPER",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    dilations=[2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_relu(self) -> None:
        self._identity_prop("Relu")

    def test_identity(self) -> None:
        self._identity_prop("Identity")

    def test_identity_sequence(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 5, 4)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("Identity", ["in_sequence"], ["output_sequence"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, None, 4)),  # type: ignore
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, None, 4)
                ),
            ],
        )  # type: ignore

    def test_identity_optional(self) -> None:
        graph = self._make_graph(
            [("in_tensor", TensorProto.FLOAT, (2, 3, 4))],
            [
                make_node("Optional", ["in_tensor"], ["in_optional"]),
                make_node("Identity", ["in_optional"], ["output_optional"]),
            ],
            [],
        )
        tensor_type_proto = helper.make_tensor_type_proto(TensorProto.FLOAT, (2, 3, 4))
        optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
        self._assert_inferred(
            graph,
            [
                helper.make_value_info("in_optional", optional_type_proto),  # type: ignore
                helper.make_value_info("output_optional", optional_type_proto),
            ],
        )  # type: ignore

    def test_identity_optional_sequence(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 5, 4)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("Optional", ["in_sequence"], ["in_optional"]),
                make_node("Identity", ["in_optional"], ["output_optional"]),
            ],
            [],
        )
        tensor_type_proto = helper.make_tensor_type_proto(
            TensorProto.FLOAT, (2, None, 4)
        )
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
        self._assert_inferred(
            graph,
            [
                helper.make_value_info("in_sequence", sequence_type_proto),  # type: ignore
                helper.make_value_info("in_optional", optional_type_proto),  # type: ignore
                helper.make_value_info("output_optional", optional_type_proto),
            ],
        )  # type: ignore

    def test_add(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 5)),
                ("y", TensorProto.FLOAT, (30, 4, 5)),
            ],
            [make_node("Add", ["x", "y"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 4, 5))]
        )

    def test_pow(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 5)),
                ("y", TensorProto.FLOAT, (30, 4, 5)),
            ],
            [make_node("Pow", ["x", "y"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (30, 4, 5))]
        )

    def test_bitshift(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT32, (2, 3, 1)),
                ("y", TensorProto.UINT32, (2, 3, 1)),
            ],
            [make_node("BitShift", ["x", "y"], "z", direction="RIGHT")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.UINT32, (2, 3, 1))]
        )

    def test_bitshift_broadcast_to_first(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT32, (16, 4, 1)), ("y", TensorProto.UINT32, (1,))],
            [make_node("BitShift", ["x", "y"], "z", direction="RIGHT")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.UINT32, (16, 4, 1))]
        )

    def test_bitshift_broadcast_to_second(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.UINT32, (1,)), ("y", TensorProto.UINT32, (2, 3, 1))],
            [make_node("BitShift", ["x", "y"], "z", direction="RIGHT")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.UINT32, (2, 3, 1))]
        )

    def test_sum_single(self) -> None:
        self._identity_prop("Sum")

    def test_sum_multi(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 4, 5)),
                ("y", TensorProto.FLOAT, (30, 4, 5)),
                ("z", TensorProto.FLOAT, (30, 4, 5)),
            ],
            [make_node("Sum", ["x", "y", "z"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (30, 4, 5))]
        )

    def test_sum_multi_broadcasting(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (30, 1, 5)),
                ("y", TensorProto.FLOAT, ("a", 4, 1)),
                ("z", TensorProto.FLOAT, (4, "b")),
            ],
            [make_node("Sum", ["x", "y", "z"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (30, 4, 5))]
        )

    def test_sum_broadcasting_param(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("a", 1, 5)),
                ("y", TensorProto.FLOAT, ("a", 4, 1)),
            ],
            [make_node("Sum", ["x", "y"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, ("a", 4, 5))]
        )

    def test_random_normal(self) -> None:
        graph = self._make_graph(
            [],
            [
                make_node(
                    "RandomNormal",
                    [],
                    ["out"],
                    dtype=TensorProto.DOUBLE,
                    shape=(3, 4, 5),
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.DOUBLE, (3, 4, 5))]
        )

    def test_random_normal_like(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("RandomNormalLike", ["X"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (2, 3, 4))]
        )

    def test_random_normal_like_with_dtype(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 3, 4))],
            [
                make_node(
                    "RandomNormalLike",
                    ["X"],
                    ["out"],
                    dtype=TensorProto.DOUBLE,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.DOUBLE, (2, 3, 4))]
        )

    def test_bernoulli(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4))],
            [make_node("Bernoulli", ["x"], ["out"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.FLOAT, (3, 4))])  # type: ignore

    def test_bernoulli_with_dtype(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4))],
            [
                make_node(
                    "Bernoulli",
                    ["x"],
                    ["out"],
                    dtype=TensorProto.DOUBLE,
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.DOUBLE, (2, 3, 4))])  # type: ignore

    def _logical_binary_op(self, op: str, input_type: TensorProto.DataType) -> None:
        graph = self._make_graph(
            [("x", input_type, (30, 4, 5)), ("y", input_type, (30, 4, 5))],
            [make_node(op, ["x", "y"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.BOOL, (30, 4, 5))]
        )

    def _logical_binary_op_with_broadcasting(
        self, op: str, input_type: TensorProto.DataType
    ) -> None:
        graph = self._make_graph(
            [("x", input_type, (1, 5)), ("y", input_type, (30, 4, 5))],
            [make_node(op, ["x", "y"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.BOOL, (30, 4, 5))]
        )

    def test_logical_and(self) -> None:
        self._logical_binary_op("And", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("And", TensorProto.BOOL)

    def test_logical_or(self) -> None:
        self._logical_binary_op("Or", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("Or", TensorProto.BOOL)

    def test_logical_xor(self) -> None:
        self._logical_binary_op("Xor", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("Xor", TensorProto.BOOL)

    def test_greater(self) -> None:
        self._logical_binary_op("Greater", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("Greater", TensorProto.BOOL)

    def test_less(self) -> None:
        self._logical_binary_op("Less", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("Less", TensorProto.BOOL)

    def test_equal(self) -> None:
        self._logical_binary_op("Equal", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("Equal", TensorProto.BOOL)

    def test_equal_string(self) -> None:
        self._logical_binary_op("Equal", TensorProto.STRING)
        self._logical_binary_op_with_broadcasting("Equal", TensorProto.STRING)

    def test_logical_not(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.BOOL, (30, 4, 5))], [make_node("Not", ["x"], "z")], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.BOOL, (30, 4, 5))]
        )

    def test_less_or_equal(self) -> None:
        self._logical_binary_op("LessOrEqual", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("LessOrEqual", TensorProto.BOOL)

    def test_greater_or_equal(self) -> None:
        self._logical_binary_op("GreaterOrEqual", TensorProto.BOOL)
        self._logical_binary_op_with_broadcasting("GreaterOrEqual", TensorProto.BOOL)

    def test_flatten(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4, 5))],
            [make_node("Flatten", ["x"], ["z"], axis=2)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (6, 20))]
        )

    def test_flatten_default_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4, 5))],
            [make_node("Flatten", ["x"], ["z"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 60))]
        )

    def test_flatten_zero_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4, 5))],
            [make_node("Flatten", ["x"], ["z"], axis=0)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (1, 120))]
        )

    def test_flatten_unknown_dim(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, "N", 4, 5))],
            [make_node("Flatten", ["x"], ["z"], axis=2)],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (None, 20))])  # type: ignore

    def test_space_to_depth(self) -> None:
        b = 10
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 100, 100))],
            [make_node("SpaceToDepth", ["x"], ["z"], blocksize=b)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 300, 10, 10))]
        )

    def test_space_to_depth_unknown_dim(self) -> None:
        b = 10
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, "N", 100, 100))],
            [make_node("SpaceToDepth", ["x"], ["z"], blocksize=b)],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, None, 10, 10))])  # type: ignore

    def test_depth_to_space(self) -> None:
        b = 10
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 300, 10, 10))],
            [make_node("DepthToSpace", ["x"], ["z"], blocksize=b, mode="DCR")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 100, 100))]
        )

    def _rnn_forward(
        self, seqlen: int, batchsize: int, inpsize: int, hiddensize: int
    ) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (seqlen, batchsize, inpsize)),
                ("w", TensorProto.FLOAT, (1, hiddensize, inpsize)),
                ("r", TensorProto.FLOAT, (1, hiddensize, hiddensize)),
            ],
            [
                make_node(
                    "RNN", ["x", "w", "r"], ["all", "last"], hidden_size=hiddensize
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "all", TensorProto.FLOAT, (seqlen, 1, batchsize, hiddensize)
                ),
                make_tensor_value_info(
                    "last", TensorProto.FLOAT, (1, batchsize, hiddensize)
                ),
            ],
        )

    def test_rnn_forward(self) -> None:
        self._rnn_forward(64, 32, 10, 4)

    def _rnn_bidirectional(
        self, seqlen: int, batchsize: int, inpsize: int, hiddensize: int
    ) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (seqlen, batchsize, inpsize)),
                ("w", TensorProto.FLOAT, (2, hiddensize, inpsize)),
                ("r", TensorProto.FLOAT, (2, hiddensize, hiddensize)),
            ],
            [
                make_node(
                    "RNN",
                    ["x", "w", "r"],
                    ["all", "last"],
                    hidden_size=hiddensize,
                    direction="bidirectional",
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "all", TensorProto.FLOAT, (seqlen, 2, batchsize, hiddensize)
                ),
                make_tensor_value_info(
                    "last", TensorProto.FLOAT, (2, batchsize, hiddensize)
                ),
            ],
        )

    def test_rnn_layout(self) -> None:
        self._rnn_layout(64, 32, 10, 4)
        self._rnn_layout(64, 32, 10, 4, "bidirectional")

    def _rnn_layout(
        self,
        seqlen: int,
        batchsize: int,
        inpsize: int,
        hiddensize: int,
        direction: str = "forward",
    ) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (batchsize, seqlen, inpsize)),
                ("w", TensorProto.FLOAT, (1, hiddensize, inpsize)),
                ("r", TensorProto.FLOAT, (1, hiddensize, hiddensize)),
            ],
            [
                make_node(
                    "RNN",
                    ["x", "w", "r"],
                    ["all", "last"],
                    hidden_size=hiddensize,
                    layout=1,
                    direction=direction,
                )
            ],
            [],
        )
        if direction == "bidirectional":
            num_directions = 2
        else:
            num_directions = 1
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "all",
                    TensorProto.FLOAT,
                    (batchsize, seqlen, num_directions, hiddensize),
                ),
                make_tensor_value_info(
                    "last", TensorProto.FLOAT, (batchsize, num_directions, hiddensize)
                ),
            ],
        )

    def test_rnn_bidirectional(self) -> None:
        self._rnn_bidirectional(64, 32, 10, 4)

    def _lstm_forward(
        self, seqlen: int, batchsize: int, inpsize: int, hiddensize: int
    ) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (seqlen, batchsize, inpsize)),
                ("w", TensorProto.FLOAT, (1, 4 * hiddensize, inpsize)),
                ("r", TensorProto.FLOAT, (1, 4 * hiddensize, hiddensize)),
            ],
            [
                make_node(
                    "LSTM",
                    ["x", "w", "r"],
                    ["all", "hidden", "last"],
                    hidden_size=hiddensize,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "all", TensorProto.FLOAT, (seqlen, 1, batchsize, hiddensize)
                ),
                make_tensor_value_info(
                    "hidden", TensorProto.FLOAT, (1, batchsize, hiddensize)
                ),
                make_tensor_value_info(
                    "last", TensorProto.FLOAT, (1, batchsize, hiddensize)
                ),
            ],
        )

    def test_lstm_forward(self) -> None:
        self._lstm_forward(64, 32, 10, 4)

    def test_topk_default_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node("TopK", ["x", "k"], ["y", "z"])],
            [],
            initializer=[make_tensor("k", TensorProto.INT64, (1,), (2,))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (3, 4, 5, 2)),
                make_tensor_value_info("z", TensorProto.INT64, (3, 4, 5, 2)),
            ],
        )

    def test_topk(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node("TopK", ["x", "k"], ["y", "z"], axis=2)],
            [],
            initializer=[make_tensor("k", TensorProto.INT64, (1,), (2,))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (3, 4, 2, 10)),
                make_tensor_value_info("z", TensorProto.INT64, (3, 4, 2, 10)),
            ],
        )

    def test_topk_raw_data(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5, 10))],
            [make_node("TopK", ["x", "k"], ["y", "z"], axis=2)],
            [],
            initializer=[
                make_tensor(
                    "k",
                    TensorProto.INT64,
                    (1,),
                    vals=np.array([3], dtype="<i8").tobytes(),
                    raw=True,
                )
            ],
        )  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (3, 4, 3, 10)),
                make_tensor_value_info("z", TensorProto.INT64, (3, 4, 3, 10)),
            ],
        )

    def test_topk_missing_k_value_output_rank_check(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5, 10)), ("k", TensorProto.INT64, (1,))],
            [make_node("TopK", ["x", "k"], ["y", "z"], axis=2)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (None, None, None, None)),  # type: ignore
                make_tensor_value_info(
                    "z", TensorProto.INT64, (None, None, None, None)
                ),
            ],
        )  # type: ignore

    def test_gemm(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (7, 5)),
                ("y", TensorProto.FLOAT, (5, 11)),
                ("z", TensorProto.FLOAT, None),
            ],
            [make_node("Gemm", ["x", "y", "z"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (7, 11))]
        )

    def test_gemm_transA(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (5, 7)),
                ("y", TensorProto.FLOAT, (5, 11)),
                ("z", TensorProto.FLOAT, None),
            ],
            [make_node("Gemm", ["x", "y", "z"], ["out"], transA=1)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (7, 11))]
        )

    def test_gemm_transB(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (7, 5)),
                ("y", TensorProto.FLOAT, (11, 5)),
                ("z", TensorProto.FLOAT, None),
            ],
            [make_node("Gemm", ["x", "y", "z"], ["out"], transB=1)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (7, 11))]
        )

    def test_gemm_transA_and_transB(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (5, 7)),
                ("y", TensorProto.FLOAT, (11, 5)),
                ("z", TensorProto.FLOAT, None),
            ],
            [make_node("Gemm", ["x", "y", "z"], ["out"], transA=1, transB=1)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (7, 11))]
        )

    def test_gemm_no_bias(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (13, 7)), ("y", TensorProto.FLOAT, (7, 17))],
            [make_node("Gemm", ["x", "y"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (13, 17))]
        )

    def test_reduce_op_shape_2_axis_opset13(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ReduceL1", "x", "y", axes=(1, 2), keepdims=0)],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (1, 2))],
        )
        operatorsetid = OperatorSetIdProto()
        operatorsetid.domain = ""
        operatorsetid.version = 13

        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (24,))],
            opset_imports=[operatorsetid],
        )

    def test_reduce_op_shape_2_axis_opset18(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11)), ("axes", TensorProto.INT64, (2,))],
            [make_node("ReduceL1", ["x", "axes"], "y", keepdims=0)],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (1, 2))],
        )
        operatorsetid = OperatorSetIdProto()
        operatorsetid.domain = ""
        operatorsetid.version = 18

        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (24,))],
            opset_imports=[operatorsetid],
        )

    def test_reduce_op_empty_set_opset13(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 0, 11))],
            [make_node("ReduceL1", "x", "y", axes=(1,), keepdims=1)],
            [],
            initializer=[],
        )
        operatorsetid = OperatorSetIdProto(domain="", version=13)
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (24, 1, 11))],
            opset_imports=[operatorsetid],
        )

    def test_reduce_op_empty_set_opset18(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 0, 11)), ("axes", TensorProto.INT64, (1,))],
            [make_node("ReduceL1", ["x", "axes"], "y", keepdims=1)],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (1,), (1,))],
        )
        operatorsetid = OperatorSetIdProto(domain="", version=18)
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (24, 1, 11))],
            opset_imports=[operatorsetid],
        )

    def test_reduce_op_shape_keep_dims_opset13(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ReduceL1", "x", "y", axes=(1, 2), keepdims=1)],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (1, 2))],
        )
        operatorsetid = OperatorSetIdProto()
        operatorsetid.domain = ""
        operatorsetid.version = 13
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (24, 1, 1))],
            opset_imports=[operatorsetid],
        )

    def test_reduce_op_shape_keep_dims_opset18(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11)), ("axes", TensorProto.INT64, (2,))],
            [make_node("ReduceL1", ["x", "axes"], "y", keepdims=1)],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (1, 2))],
        )
        operatorsetid = OperatorSetIdProto()
        operatorsetid.domain = ""
        operatorsetid.version = 18
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (24, 1, 1))],
            opset_imports=[operatorsetid],
        )

    def test_reduce_op_shape_default_value(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ReduceL1", "x", "y")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 1, 1))]
        )

    def test_reduce_op_shape_no_axes_do_not_keep_dims(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ReduceL1", "x", "y", keepdims=0)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, ())]
        )

    def test_reduce_op_shape_negative_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11)), ("axes", TensorProto.INT64, (2,))],
            [make_node("ReduceL1", ["x", "axes"], "y")],
            [],
            initializer=[make_tensor("axes", TensorProto.INT64, (2,), (-1, -2))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (24, 1, 1))]
        )

    def test_argmax_shape(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ArgMax", "x", "y", axis=1, keepdims=1)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT64, (24, 1, 11))]
        )

    def test_argmax_shape_keepdims(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ArgMax", "x", "y", axis=0, keepdims=0)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT64, (4, 11))]
        )

    def test_argmax_shape_default_value(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))], [make_node("ArgMax", "x", "y")], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT64, (1, 4, 11))]
        )

    def test_argmax_shape_negative_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (24, 4, 11))],
            [make_node("ArgMax", "x", "y", axis=-2)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT64, (24, 1, 11))]
        )

    def test_dropout(self) -> None:
        graph = self._make_graph(
            [
                (
                    "data",
                    TensorProto.FLOAT,
                    (
                        3,
                        4,
                        5,
                    ),
                ),
                ("ratio", TensorProto.FLOAT, ()),
            ],
            [make_node("Dropout", ["data", "ratio"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "out",
                    TensorProto.FLOAT,
                    (
                        3,
                        4,
                        5,
                    ),
                )
            ],
        )

    def test_LRN(self) -> None:
        self._identity_prop("LRN", alpha=0.5, beta=0.5, size=1)

    def test_batch_norm(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5, 6, 7)),
                ("scale", TensorProto.FLOAT, (4,)),
                ("b", TensorProto.FLOAT, (4,)),
                ("mean", TensorProto.FLOAT, (4,)),
                ("var", TensorProto.FLOAT, (4,)),
            ],
            [
                make_node(
                    "BatchNormalization", ["x", "scale", "b", "mean", "var"], ["out"]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (3, 4, 5, 6, 7))]
        )

    def test_batch_norm_rank1(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (128,)),  # 1-dimensional permitted
                ("scale", TensorProto.FLOAT, (1,)),
                ("b", TensorProto.FLOAT, (1,)),
                ("mean", TensorProto.FLOAT, (1,)),
                ("var", TensorProto.FLOAT, (1,)),
            ],
            [
                make_node(
                    "BatchNormalization", ["x", "scale", "b", "mean", "var"], ["out"]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (128,))]
        )

    def test_batch_norm_invalid(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (128,)),
                ("scale", TensorProto.FLOAT, (1, 2)),  # invalid rank
                ("b", TensorProto.FLOAT, (1,)),
                ("mean", TensorProto.FLOAT, (1,)),
                ("var", TensorProto.FLOAT, (1,)),
            ],
            [
                make_node(
                    "BatchNormalization", ["x", "scale", "b", "mean", "var"], ["out"]
                )
            ],
            [],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    def test_split_negative_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4))],
            [make_node("Split", ["x"], ["y", "z"], axis=-1, num_outputs=2)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (2, 2)),
                make_tensor_value_info("z", TensorProto.FLOAT, (2, 2)),
            ],
        )

    def test_split_with_split_attribute(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 4)), ("split", TensorProto.INT64, (2,))],
            [make_node("Split", ["x", "split"], ["y", "z"], axis=1)],
            [],
            initializer=[make_tensor("split", TensorProto.INT64, (2,), (3, 1))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (2, 3)),
                make_tensor_value_info("z", TensorProto.FLOAT, (2, 1)),
            ],
        )

    def test_split_with_split_attribute_unknown_split_dim(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (2, "a", "b")),
                ("split", TensorProto.INT64, (2,)),
            ],
            [make_node("Split", ["x", "split"], ["y", "z"], axis=1)],
            [],
            initializer=[make_tensor("split", TensorProto.INT64, (2,), (3, 1))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (2, None, "b")),  # type: ignore
                make_tensor_value_info("z", TensorProto.FLOAT, (2, None, "b")),
            ],
        )  # type: ignore

    def test_split_from_GLU(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (5, 6, 7))],
            [make_node("Split", ["x"], ["y", "z"], axis=1, num_outputs=2)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (5, 3, 7)),
                make_tensor_value_info("z", TensorProto.FLOAT, (5, 3, 7)),
            ],
        )

    def test_split_uneven_split_2d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (8, 2))],
            [make_node("Split", ["x"], ["y", "z", "a"], axis=0, num_outputs=3)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (3, 2)),
                make_tensor_value_info("z", TensorProto.FLOAT, (3, 2)),
                make_tensor_value_info("a", TensorProto.FLOAT, (2, 2)),
            ],
        )

    def test_split_uneven_split_3d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 7, 3))],
            [make_node("Split", ["x"], ["y", "z", "a"], axis=1, num_outputs=3)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (2, 3, 3)),
                make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 3)),
                make_tensor_value_info("a", TensorProto.FLOAT, (2, 1, 3)),
            ],
        )

    def test_GLU_partial(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (5, 6, 7))],
            [
                make_node("Split", ["x"], ["y", "z"], axis=1, num_outputs=2),
                make_node("Sigmoid", ["z"], ["a"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (5, 3, 7)),
                make_tensor_value_info("z", TensorProto.FLOAT, (5, 3, 7)),
                make_tensor_value_info("a", TensorProto.FLOAT, (5, 3, 7)),
            ],
        )

    def test_GLU(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (5, 6, 7))],
            [
                make_node("Split", ["x"], ["y", "z"], axis=1, num_outputs=2),
                make_node("Sigmoid", ["z"], ["a"]),
                make_node("Mul", ["y", "a"], ["b"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.FLOAT, (5, 3, 7)),
                make_tensor_value_info("z", TensorProto.FLOAT, (5, 3, 7)),
                make_tensor_value_info("a", TensorProto.FLOAT, (5, 3, 7)),
                make_tensor_value_info("b", TensorProto.FLOAT, (5, 3, 7)),
            ],
        )

    def test_softmax_2d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5))], [make_node("Softmax", ["x"], "z")], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5))]
        )

    def test_softmax_3d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6))],
            [make_node("Softmax", ["x"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5, 6))]
        )

    def test_hardmax_2d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5))], [make_node("Hardmax", ["x"], "z")], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5))]
        )

    def test_hardmax_3d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6))],
            [make_node("Hardmax", ["x"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5, 6))]
        )

    def test_logsoftmax_2d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5))],
            [make_node("LogSoftmax", ["x"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5))]
        )

    def test_logsoftmax_3d(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6))],
            [make_node("LogSoftmax", ["x"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5, 6))]
        )

    def test_logsoftmax_3d_negative_axis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6))],
            [make_node("LogSoftmax", ["x"], "z", axis=-1)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5, 6))]
        )

    def test_maxpool(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))]
        )

    def test_maxpool_with_indices(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y", "Z"], kernel_shape=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3)),
                make_tensor_value_info("Z", TensorProto.INT64, (5, 3, 3, 3)),
            ],
        )

    def test_maxpool_3D(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3, 3))]
        )

    def test_maxpool_with_padding(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 6, 6))]
        )

    def test_maxpool_with_padding_and_stride(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[2, 2],
                    pads=[1, 1, 2, 2],
                    strides=[2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))]
        )

    def test_maxpool_with_floor_mode(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (32, 288, 35, 35))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    ceil_mode=False,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (32, 288, 17, 17))]
        )

    def test_maxpool_with_ceil_mode(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (32, 288, 35, 35))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    ceil_mode=True,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (32, 288, 18, 18))]
        )

    def test_maxpool_ceil(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (1, 1, 4, 4))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[3, 3],
                    strides=[2, 2],
                    ceil_mode=True,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (1, 1, 2, 2))]
        )

    def test_maxpool_with_dilations(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("MaxPool", ["X"], ["Y"], kernel_shape=[2, 2], dilations=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_maxpool_with_same_upper_padding_and_stride(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_UPPER",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_maxpool_with_same_upper_padding_and_stride_and_dilation(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_UPPER",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    dilations=[2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_maxpool_with_same_upper_padding_and_stride_one(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_UPPER",
                    kernel_shape=[2, 2],
                    strides=[1, 1],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 4, 4))]
        )

    def test_maxpool_with_same_lower_padding_and_stride(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 9, 9))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_LOWER",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 5, 5))]
        )

    def test_maxpool_with_same_lower_padding_and_stride_and_dilation(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 9, 9))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_LOWER",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    dilations=[2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 5, 5))]
        )

    def test_maxpool_with_same_lower_padding_and_big_stride(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "MaxPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_LOWER",
                    kernel_shape=[2, 2],
                    strides=[4, 4],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))]
        )

    def test_averagepool(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))]
        )

    def test_averagepool_3D(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4, 4))],
            [make_node("AveragePool", ["X"], ["Y"], kernel_shape=[2, 2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3, 3))]
        )

    def test_averagepool_with_padding(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "AveragePool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 6, 6))]
        )

    def test_averagepool_with_padding_and_stride(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "AveragePool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[2, 2],
                    pads=[1, 1, 2, 2],
                    strides=[2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))]
        )

    def test_averagepool_ceil(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (1, 1, 4, 4))],
            [
                make_node(
                    "AveragePool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[3, 3],
                    strides=[2, 2],
                    ceil_mode=True,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (1, 1, 2, 2))]
        )

    def test_lppool(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))]
        )

    def test_lppool_3D(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3, 3))]
        )

    def test_lppool_with_padding(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2], pads=[1, 1, 2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 6, 6))]
        )

    def test_lppool_with_padding_and_stride(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "LpPool",
                    ["X"],
                    ["Y"],
                    kernel_shape=[2, 2],
                    pads=[1, 1, 2, 2],
                    strides=[2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 3, 3))]
        )

    def test_lppool_with_dilations(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("LpPool", ["X"], ["Y"], kernel_shape=[2, 2], dilations=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_lppool_with_same_upper_padding_and_stride_and_dilation(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [
                make_node(
                    "LpPool",
                    ["X"],
                    ["Y"],
                    auto_pad="SAME_UPPER",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                    dilations=[2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 2, 2))]
        )

    def test_roipool(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (5, 3, 4, 4)),
                ("rois", TensorProto.INT64, (2, 5)),
            ],
            [make_node("MaxRoiPool", ["X", "rois"], ["Y"], pooled_shape=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (2, 3, 2, 2))]
        )

    def test_lp_norm(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5, 6, 7))],
            [make_node("LpNormalization", ["x"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (3, 4, 5, 6, 7))]
        )

    def test_instance_norm(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5, 6, 7)),
                ("scale", TensorProto.FLOAT, (4,)),
                ("b", TensorProto.FLOAT, (4,)),
            ],
            [make_node("InstanceNormalization", ["x", "scale", "b"], ["out"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("out", TensorProto.FLOAT, (3, 4, 5, 6, 7))]
        )

    def test_global_maxpool(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("GlobalMaxPool", ["X"], ["Y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))]
        )

    def test_global_averagepool(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("GlobalAveragePool", ["X"], ["Y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))]
        )

    def test_global_lppool(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (5, 3, 4, 4))],
            [make_node("GlobalLpPool", ["X"], ["Y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (5, 3, 1, 1))]
        )

    def test_conv_transpose(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [make_node("ConvTranspose", ["X", "W"], "Y", strides=[2, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 32, 33, 33))]
        )

    def test_conv_transpose_with_pads(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [
                make_node(
                    "ConvTranspose", ["X", "W"], "Y", strides=[2, 2], pads=[1, 1, 2, 2]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 32, 30, 30))]
        )

    def test_conv_transpose_with_output_shape(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W"],
                    "Y",
                    strides=[2, 2],
                    pads=[1, 1, 2, 2],
                    output_shape=[36, 36],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 32, 36, 36))]
        )

    def test_conv_transpose_with_kernel_shape(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, None, None)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W"],
                    "Y",
                    kernel_shape=[3, 3],
                    strides=[2, 2],
                    pads=[1, 1, 2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 32, 30, 30))]
        )

    def test_conv_transpose_with_dilations(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W"],
                    "Y",
                    strides=[2, 2],
                    pads=[1, 1, 2, 2],
                    dilations=[3, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 32, 34, 34))]
        )

    def test_conv_transpose_with_group(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W"],
                    "Y",
                    strides=[2, 2],
                    pads=[1, 1, 2, 2],
                    group=2,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 64, 30, 30))]
        )

    def test_conv_transpose_with_group_and_output_shape(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W"],
                    "Y",
                    strides=[2, 2],
                    pads=[1, 1, 2, 2],
                    group=2,
                    output_shape=[36, 36],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 64, 36, 36))]
        )

    def test_conv_transpose_with_pads_and_auto_pads(self) -> None:
        # This test should fail because pads cannot be used simultaneously with auto_pad
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (1, 1, 2, 2)),
                ("W", TensorProto.FLOAT, (1, 1, 3, 3)),
                ("B", TensorProto.FLOAT, (1,)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W", "B"],
                    "Y",
                    auto_pad="SAME_UPPER",
                    strides=[1, 1],
                    pads=[0, 1, 1, 0],
                )
            ],
            [],
        )
        self.assertRaises(
            onnx.shape_inference.InferenceError,
            onnx.shape_inference.infer_shapes,
            helper.make_model(graph),
            strict_mode=True,
        )

    def test_conv_transpose_auto_pads(self) -> None:
        graph = self._make_graph(
            [
                ("X", TensorProto.FLOAT, (25, 48, 16, 16)),
                ("W", TensorProto.FLOAT, (48, 32, 3, 3)),
            ],
            [
                make_node(
                    "ConvTranspose",
                    ["X", "W"],
                    "Y",
                    auto_pad="SAME_UPPER",
                    strides=[2, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 32, 32, 32))]
        )

    def test_mvn_function_output_shape(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (25, 48, 16, 16))],
            [make_node("MeanVarianceNormalization", "X", "Y", axes=[0, 2, 3])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 48, 16, 16))]
        )

    def test_scan(self) -> None:
        batch_size = 1
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Scan to match
        # the GraphProto, but Scan knows nothing about the additional inputs.
        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.UNDEFINED, None),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("loop_state_orig", TensorProto.FLOAT, (batch_size, loop_state_size)),
                ("scan_input", TensorProto.FLOAT, (batch_size, seq_len, input_size)),
            ],
            [
                make_node(
                    "Scan",
                    ["", "loop_state_orig", "scan_input"],
                    ["loop_state_final", "scan_output"],
                    num_scan_inputs=1,
                    body=subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "loop_state_final", TensorProto.FLOAT, (batch_size, loop_state_size)
                ),
                make_tensor_value_info(
                    "scan_output", TensorProto.FLOAT, (batch_size, seq_len, input_size)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 8)],
        )

    def test_scan_opset9(self) -> None:
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Scan to match
        # the GraphProto, but Scan knows nothing about the additional inputs.
        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.UNDEFINED, None),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("loop_state_orig", TensorProto.FLOAT, (loop_state_size,)),
                ("scan_input", TensorProto.FLOAT, (seq_len, input_size)),
            ],
            [
                make_node(
                    "Scan",
                    ["loop_state_orig", "scan_input"],
                    ["loop_state_final", "scan_output"],
                    num_scan_inputs=1,
                    body=subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "loop_state_final", TensorProto.FLOAT, (loop_state_size,)
                ),
                make_tensor_value_info(
                    "scan_output", TensorProto.FLOAT, (seq_len, input_size)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)],
        )

    def test_scan_opset9_axes(self) -> None:
        axis_0_len = "axis0"
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Scan to match
        # the GraphProto, but Scan knows nothing about the additional inputs.
        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.UNDEFINED, None),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("loop_state_orig", TensorProto.FLOAT, (loop_state_size,)),
                ("scan_input", TensorProto.FLOAT, (axis_0_len, seq_len, input_size)),
            ],
            [
                make_node(
                    "Scan",
                    ["loop_state_orig", "scan_input"],
                    ["loop_state_final", "scan_output"],
                    num_scan_inputs=1,
                    body=subgraph,
                    scan_input_axes=[1],
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "loop_state_final", TensorProto.FLOAT, (loop_state_size,)
                ),
                make_tensor_value_info(
                    "scan_output", TensorProto.FLOAT, (seq_len, axis_0_len, input_size)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)],
        )

    def test_scan_opset9_output_axes(self) -> None:
        axis_0_len = "axis0"
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.UNDEFINED, None),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("loop_state_orig", TensorProto.FLOAT, (loop_state_size,)),
                ("scan_input", TensorProto.FLOAT, (axis_0_len, seq_len, input_size)),
            ],
            [
                make_node(
                    "Scan",
                    ["loop_state_orig", "scan_input"],
                    ["loop_state_final", "scan_output"],
                    num_scan_inputs=1,
                    body=subgraph,
                    scan_input_axes=[1],
                    scan_output_axes=[1],
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "loop_state_final", TensorProto.FLOAT, (loop_state_size,)
                ),
                make_tensor_value_info(
                    "scan_output", TensorProto.FLOAT, (axis_0_len, seq_len, input_size)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)],
        )

    def test_scan_opset9_negative_axes(self) -> None:
        axis_0_len = "axis0"
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.UNDEFINED, None),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("loop_state_orig", TensorProto.FLOAT, (loop_state_size,)),
                ("scan_input", TensorProto.FLOAT, (axis_0_len, seq_len, input_size)),
            ],
            [
                make_node(
                    "Scan",
                    ["loop_state_orig", "scan_input"],
                    ["loop_state_final", "scan_output"],
                    num_scan_inputs=1,
                    body=subgraph,
                    scan_input_axes=[-2],
                    scan_output_axes=[-2],
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "loop_state_final", TensorProto.FLOAT, (loop_state_size,)
                ),
                make_tensor_value_info(
                    "scan_output", TensorProto.FLOAT, (axis_0_len, seq_len, input_size)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 9)],
        )

    def test_if_ver1(self) -> None:
        # Create a simple If node where the 'then' subgraph adds to the current value, and the 'else' subgraph
        # subtracts.
        # can't use self._make_graph for the subgraphs as that add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the subgraphs to have zero inputs
        then_subgraph = helper.make_graph(
            [make_node("Add", ["current_value", "add_value"], ["then_output"])],
            "then_subgraph",
            [],  # no inputs
            [make_tensor_value_info("then_output", TensorProto.UNDEFINED, None)],
        )

        else_subgraph = helper.make_graph(
            [make_node("Sub", ["current_value", "sub_value"], ["else_output"])],
            "else_subgraph",
            [],  # no inputs
            [make_tensor_value_info("else_output", TensorProto.UNDEFINED, None)],
        )

        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (1,)),
                ("current_value", TensorProto.FLOAT, (1,)),
                ("add_value", TensorProto.FLOAT, (1,)),
                ("sub_value", TensorProto.FLOAT, (1,)),
            ],
            [
                make_node(
                    "If",
                    ["cond"],
                    ["if_output"],
                    then_branch=then_subgraph,
                    else_branch=else_subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info("if_output", TensorProto.FLOAT, (1,))],
            opset_imports=[make_opsetid(ONNX_DOMAIN, 10)],
        )

    def test_if(self) -> None:
        # Create a simple If node where the 'then' subgraph adds to the current value, and the 'else' subgraph
        # subtracts.
        # can't use self._make_graph for the subgraphs as that add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the subgraphs to have zero inputs
        then_subgraph = helper.make_graph(
            [make_node("Add", ["current_value", "add_value"], ["then_output"])],
            "then_subgraph",
            [],  # no inputs
            [make_tensor_value_info("then_output", TensorProto.UNDEFINED, None)],
        )

        else_subgraph = helper.make_graph(
            [make_node("Sub", ["current_value", "sub_value"], ["else_output"])],
            "else_subgraph",
            [],  # no inputs
            [make_tensor_value_info("else_output", TensorProto.UNDEFINED, None)],
        )

        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (1,)),
                ("current_value", TensorProto.FLOAT, (1,)),
                ("add_value", TensorProto.FLOAT, (1,)),
                ("sub_value", TensorProto.FLOAT, (1,)),
            ],
            [
                make_node(
                    "If",
                    ["cond"],
                    ["if_output"],
                    then_branch=then_subgraph,
                    else_branch=else_subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph, [make_tensor_value_info("if_output", TensorProto.FLOAT, (1,))]
        )

    def test_if_with_different_shapes_in_then_else_branches(self) -> None:
        # Create a simple If node where the 'then' subgraph adds to the current value, and the 'else' subgraph
        # subtracts.
        # can't use self._make_graph for the subgraphs as that add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the subgraphs to have zero inputs
        then_subgraph = helper.make_graph(
            [make_node("Add", ["current_value", "add_value"], ["then_output"])],
            "then_subgraph",
            [],  # no inputs
            [make_tensor_value_info("then_output", TensorProto.UNDEFINED, (1,))],
        )

        else_subgraph = helper.make_graph(
            [make_node("Sub", ["current_value", "sub_value"], ["else_output"])],
            "else_subgraph",
            [],  # no inputs
            [make_tensor_value_info("else_output", TensorProto.UNDEFINED, (5,))],
        )

        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (1,)),
                ("current_value", TensorProto.FLOAT, (1,)),
                ("add_value", TensorProto.FLOAT, (1,)),
                ("sub_value", TensorProto.FLOAT, (5,)),
            ],
            [
                make_node(
                    "If",
                    ["cond"],
                    ["if_output"],
                    then_branch=then_subgraph,
                    else_branch=else_subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(graph, [make_tensor_value_info("if_output", TensorProto.FLOAT, (None,))])  # type: ignore

    def test_if_no_shape_in_then_branch(self) -> None:
        then_graph = parse_graph(
            "then_graph () => (then_output) { then_output = ReduceSum <keepdims=0> (X, axes) }"
        )
        else_graph = parse_graph(
            "else_graph () => (else_output) { else_output = ReduceSum <keepdims=0> (X) }"
        )
        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (1,)),
                ("X", TensorProto.FLOAT, (4, 8, 16)),
                ("axes", TensorProto.INT64, (1,)),
            ],
            [
                make_node(
                    "If",
                    ["cond"],
                    ["if_output"],
                    then_branch=then_graph,
                    else_branch=else_graph,
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("if_output", TensorProto.FLOAT, None)])  # type: ignore

    def test_if_no_shape_in_else_branch(self) -> None:
        then_graph = parse_graph(
            "then_graph () => (then_output) { then_output = ReduceSum <keepdims=0> (X) }"
        )
        else_graph = parse_graph(
            "else_graph () => (else_output) { else_output = ReduceSum <keepdims=0> (X, axes) }"
        )
        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (1,)),
                ("X", TensorProto.FLOAT, (4, 8, 16)),
                ("axes", TensorProto.INT64, (1,)),
            ],
            [
                make_node(
                    "If",
                    ["cond"],
                    ["if_output"],
                    then_branch=then_graph,
                    else_branch=else_graph,
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("if_output", TensorProto.FLOAT, None)])  # type: ignore

    def test_if_with_different_optional_shapes_in_then_else_branches(self) -> None:
        # Create a simple If node where the 'then' subgraph adds to the current value, and the 'else' subgraph
        # subtracts.
        # can't use self._make_graph for the subgraphs as that add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the subgraphs to have zero inputs
        then_tensor_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.UNDEFINED,
            shape=[
                1,
            ],
        )
        then_optional_type_proto = helper.make_optional_type_proto(then_tensor_proto)
        then_optional_vi = helper.make_value_info(
            "then_optional_output", then_optional_type_proto
        )
        then_subgraph = helper.make_graph(
            [make_node("Optional", ["then_tensor_value"], ["then_optional_output"])],
            "then_subgraph",
            [],  # no inputs
            [then_optional_vi],
        )

        else_tensor_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.UNDEFINED,
            shape=[
                5,
            ],
        )
        else_optional_type_proto = helper.make_optional_type_proto(else_tensor_proto)
        else_optional_vi = helper.make_value_info(
            "else_optional_output", else_optional_type_proto
        )
        else_subgraph = helper.make_graph(
            [make_node("Optional", ["else_tensor_value"], ["else_optional_output"])],
            "else_subgraph",
            [],  # no inputs
            [else_optional_vi],
        )

        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (1,)),
                ("then_tensor_value", TensorProto.FLOAT, (1,)),
                ("else_tensor_value", TensorProto.FLOAT, (5,)),
            ],
            [
                make_node(
                    "If",
                    ["cond"],
                    ["if_output"],
                    then_branch=then_subgraph,
                    else_branch=else_subgraph,
                )
            ],
            [],
        )

        output_tensor_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=(None,)
        )
        output_optional_type_proto = helper.make_optional_type_proto(
            output_tensor_proto
        )
        output_optional_vi = helper.make_value_info(
            "if_output", output_optional_type_proto
        )
        self._assert_inferred(graph, [output_optional_vi])  # type: ignore

    def test_maxunpool_shape_without_output_shape(self) -> None:
        graph = self._make_graph(
            [
                ("xT", TensorProto.FLOAT, (1, 1, 2, 2)),
                ("xI", TensorProto.FLOAT, (1, 1, 2, 2)),
            ],
            [
                make_node(
                    "MaxUnpool", ["xT", "xI"], "Y", kernel_shape=[2, 2], strides=[2, 2]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (1, 1, 4, 4))]
        )

    def test_maxunpool_shape_with_output_shape(self) -> None:
        graph = self._make_graph(
            [
                ("xT", TensorProto.FLOAT, (1, 1, 2, 2)),
                ("xI", TensorProto.FLOAT, (1, 1, 2, 2)),
                ("output_shape", TensorProto.FLOAT, (4,)),
            ],
            [
                make_node(
                    "MaxUnpool",
                    ["xT", "xI", "output_shape"],
                    "Y",
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                )
            ],
            [make_tensor_value_info("Y", TensorProto.FLOAT, None)],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, None)]
        )

    def test_onehot_without_axis(self) -> None:
        graph = self._make_graph(
            [
                ("indices", TensorProto.INT64, (2, 2)),
                ("depth", TensorProto.INT64, ()),
                ("values", TensorProto.FLOAT, (2,)),
            ],
            [make_node("OneHot", ["indices", "depth", "values"], "Y")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2, None))])  # type: ignore

    def test_onehot_with_axis(self) -> None:
        graph = self._make_graph(
            [
                ("indices", TensorProto.INT64, (2, 3, 5)),
                ("depth", TensorProto.INT64, (1,)),
                ("values", TensorProto.FLOAT, (2,)),
            ],
            [make_node("OneHot", ["indices", "depth", "values"], "Y", axis=1)],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (2, None, 3, 5))])  # type: ignore

    def test_onehot_without_axis_2(self) -> None:
        graph = self._make_graph(
            [
                ("indices", TensorProto.INT64, (2, 2)),
                ("depth", TensorProto.INT64, ()),
                ("values", TensorProto.FLOAT, (2,)),
            ],
            [make_node("OneHot", ["indices", "depth", "values"], "Y")],
            [],
            initializer=[make_tensor("depth", TensorProto.INT64, (), (256,))],
        )
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (2, 2, 256))])  # type: ignore

    def test_onehot_with_axis_2(self) -> None:
        graph = self._make_graph(
            [
                ("indices", TensorProto.INT64, (2, 3, 5)),
                ("depth", TensorProto.INT64, (1,)),
                ("values", TensorProto.FLOAT, (2,)),
            ],
            [make_node("OneHot", ["indices", "depth", "values"], "Y", axis=1)],
            [],
            initializer=[make_tensor("depth", TensorProto.INT64, (1,), (256,))],
        )
        self._assert_inferred(graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (2, 256, 3, 5))])  # type: ignore

    def test_loop(self) -> None:
        # can't use self._make_graph for the subgraph as it add more inputs for the Reshape operations it inserts.
        # this breaks the subgraph inferencing as it expects the number of inputs passed from Loop to match
        # the GraphProto, but Loop knows nothing about the additional inputs.
        input_value_infos = [
            make_tensor_value_info("iter_num_in", TensorProto.INT64, (1,)),
            make_tensor_value_info("cond_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, ()),
        ]
        output_value_infos = [
            make_tensor_value_info("cond_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.FLOAT, (3,)),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["cond_in"], ["cond_out"]),
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Identity", ["outer_scope_input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("max_trip_count", TensorProto.INT64, (1,)),
                ("cond_orig", TensorProto.FLOAT, (1,)),
                ("loop_state_orig", TensorProto.FLOAT, (2,)),
                ("outer_scope_input", TensorProto.FLOAT, (3,)),
            ],
            [
                make_node(
                    "Loop",
                    ["max_trip_count", "cond_orig", "loop_state_orig"],
                    ["loop_state_final", "loop_output"],
                    body=subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "loop_state_final", TensorProto.FLOAT, None
                ),  # shape may change between iterations
                make_tensor_value_info("loop_output", TensorProto.FLOAT, (None, 3)),
            ],
        )  # type: ignore

    def test_loop_no_state(self) -> None:
        input_value_infos = [
            make_tensor_value_info("iter_num_in", TensorProto.INT64, (1,)),
            make_tensor_value_info("cond_in", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("cond_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.FLOAT, (3,)),
        ]

        subgraph = helper.make_graph(
            [
                make_node("Identity", ["cond_in"], ["cond_out"]),
                make_node("Identity", ["outer_scope_input"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        graph = self._make_graph(
            [
                ("max_trip_count", TensorProto.INT64, (1,)),
                ("cond_orig", TensorProto.FLOAT, (1,)),
                ("outer_scope_input", TensorProto.FLOAT, (3,)),
            ],
            [
                make_node(
                    "Loop",
                    ["max_trip_count", "cond_orig"],
                    ["loop_output"],
                    body=subgraph,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph, [make_tensor_value_info("loop_output", TensorProto.FLOAT, (None, 3))]
        )  # type: ignore

    def test_constantofshape_with_input_shape(self) -> None:
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (3,), (3, 4, 5)),
                ),
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
        )  # type: ignore

    def test_constantofshape_without_input_shape(self) -> None:
        graph = self._make_graph(
            [("shape", TensorProto.INT64, (3,))],
            [
                make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["y"],
                    value=make_tensor("value", TensorProto.UINT8, (1,), (2,)),
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (None, None, None))]
        )  # type: ignore

    def test_constantofshape_without_input_shape_scalar(self) -> None:
        graph = self._make_graph(
            [("shape", TensorProto.INT64, (0,))],
            [
                make_node(
                    "ConstantOfShape",
                    ["shape"],
                    ["y"],
                    value=make_tensor("value", TensorProto.UINT8, (1,), (2,)),
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, ())]
        )  # type: ignore

    def test_constantofshape_with_shape_zero(self) -> None:
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (1,), (0,)),
                ),
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
                make_tensor_value_info("shape", TensorProto.INT64, (1,)),
                make_tensor_value_info("y", TensorProto.INT32, (0,)),
            ],
        )  # type: ignore

    def test_convinteger(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (3, 4, 5, 6, 7)),
                ("y", TensorProto.UINT8, (5, 4, 2, 4, 3)),
            ],
            [
                make_node(
                    "ConvInteger",
                    ["x", "y"],
                    "z",
                    pads=[0, 1, 1, 0, 0, 1],
                    dilations=[1, 2, 2],
                    strides=[1, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.INT32, (3, 5, 4, 1, 3))]
        )

    def test_convinetger_dilations(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 8, 8, 8)),
                ("y", TensorProto.INT8, (50, 4, 3, 3, 3)),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "ConvInteger",
                    ["x", "y", "x_zero_point", "y_zero_point"],
                    "z",
                    dilations=[1, 2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.INT32, (30, 50, 6, 4, 2))]
        )

    def test_convinteger_strides(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.INT8, (30, 4, 8, 8, 8)),
                ("y", TensorProto.INT8, (50, 4, 3, 3, 3)),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "ConvInteger",
                    ["x", "y", "x_zero_point", "y_zero_point"],
                    "z",
                    strides=[1, 2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.INT32, (30, 50, 6, 3, 2))]
        )

    def test_convineteger_pads(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 7, 6, 4)),
                ("y", TensorProto.INT8, (50, 4, 3, 3, 3)),
            ],
            [make_node("ConvInteger", ["x", "y"], "z", pads=[1, 1, 2, 0, 1, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.INT32, (30, 50, 6, 6, 6))]
        )

    def test_convineteger_group(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.INT8, (30, 4, 8, 8, 8)),
                ("y", TensorProto.INT8, (4, 1, 8, 8, 8)),
            ],
            [make_node("ConvInteger", ["x", "y"], "z", group=4)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.INT32, (30, 4, 1, 1, 1))]
        )

    def test_convineteger_partial_missing_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, None, 6, 4)),
                ("y", TensorProto.UINT8, (50, 4, 3, 3, 3)),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "ConvInteger",
                    ["x", "y", "x_zero_point", "y_zero_point"],
                    "z",
                    pads=[1, 1, 2, 0, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.INT32, (30, 50, None, 6, 6))])  # type: ignore

    def test_convineteger_partial_missing_weight_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 7, 6, 4)),
                ("y", TensorProto.UINT8, (50, 4, None, 3, 3)),
            ],
            [make_node("ConvInteger", ["x", "y"], "z", pads=[1, 1, 2, 0, 1, 2])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.INT32, None)]
        )

    def test_qlinearconv(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (3, 4, 5, 6, 7)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("w", TensorProto.UINT8, (5, 4, 2, 4, 3)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.UINT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    pads=[0, 1, 1, 0, 0, 1],
                    dilations=[1, 2, 2],
                    strides=[1, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (3, 5, 4, 1, 3))]
        )

    def test_qlinearconv_dilations(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 8, 8, 8)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("w", TensorProto.UINT8, (50, 4, 3, 3, 3)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.UINT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    dilations=[1, 2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (30, 50, 6, 4, 2))]
        )

    def test_qlinearconv_strides(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.INT8, (30, 4, 8, 8, 8)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.INT8, ()),
                ("w", TensorProto.INT8, (50, 4, 3, 3, 3)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.INT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.INT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    strides=[1, 2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT8, (30, 50, 6, 3, 2))]
        )

    def test_qlinearconv_pads(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 7, 6, 4)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("w", TensorProto.INT8, (50, 4, 3, 3, 3)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.INT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    pads=[1, 1, 2, 0, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (30, 50, 6, 6, 6))]
        )

    def test_qlinearconv_group(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.INT8, (30, 4, 8, 8, 8)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.INT8, ()),
                ("w", TensorProto.INT8, (4, 1, 8, 8, 8)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.INT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.INT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    group=4,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT8, (30, 4, 1, 1, 1))]
        )

    def test_qlinearconv_partial_missing_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, None, 6, 4)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("w", TensorProto.UINT8, (50, 4, 3, 3, 3)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.UINT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    pads=[1, 1, 2, 0, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.UINT8, (30, 50, None, 6, 6))])  # type: ignore

    def test_qlinearconv_partial_missing_weight_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 7, 6, 4)),
                ("x_scale", TensorProto.FLOAT, ()),
                ("x_zero_point", TensorProto.UINT8, ()),
                ("w", TensorProto.UINT8, (50, 4, None, 3, 3)),
                ("w_scale", TensorProto.FLOAT, ()),
                ("w_zero_point", TensorProto.UINT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearConv",
                    [
                        "x",
                        "x_scale",
                        "x_zero_point",
                        "w",
                        "w_scale",
                        "w_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    "y",
                    pads=[1, 1, 2, 0, 1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, None)]
        )

    def _make_qlinearmatmul_test(
        self, shape1: Sequence[int], shape2: Sequence[int]
    ) -> None:
        expected_out_shape = np.matmul(
            np.arange(np.prod(shape1)).reshape(shape1),
            np.arange(np.prod(shape2)).reshape(shape2),
        ).shape
        graph = self._make_graph(
            [
                ("a", TensorProto.UINT8, shape1),
                ("a_scale", TensorProto.FLOAT, ()),
                ("a_zero_point", TensorProto.UINT8, ()),
                ("b", TensorProto.UINT8, shape2),
                ("b_scale", TensorProto.FLOAT, ()),
                ("b_zero_point", TensorProto.UINT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearMatMul",
                    [
                        "a",
                        "a_scale",
                        "a_zero_point",
                        "b",
                        "b_scale",
                        "b_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    ["y"],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, expected_out_shape)]
        )

    def test_qlinearmatmul(self) -> None:
        self._make_qlinearmatmul_test((3,), (3,))
        self._make_qlinearmatmul_test((4, 2), (2, 4))
        self._make_qlinearmatmul_test((2,), (2, 3))
        self._make_qlinearmatmul_test((4, 2), (2,))
        self._make_qlinearmatmul_test((5, 1, 4, 2), (1, 3, 2, 3))
        self._make_qlinearmatmul_test((4, 2), (3, 2, 3))

    def _make_qlinearmatmul_test_allow_unknown(
        self, shape1: Any, shape2: Any, expected_out_shape: Any
    ) -> None:
        graph = self._make_graph(
            [
                ("a", TensorProto.UINT8, shape1),
                ("a_scale", TensorProto.FLOAT, ()),
                ("a_zero_point", TensorProto.UINT8, ()),
                ("b", TensorProto.UINT8, shape2),
                ("b_scale", TensorProto.FLOAT, ()),
                ("b_zero_point", TensorProto.UINT8, ()),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "QLinearMatMul",
                    [
                        "a",
                        "a_scale",
                        "a_zero_point",
                        "b",
                        "b_scale",
                        "b_zero_point",
                        "y_scale",
                        "y_zero_point",
                    ],
                    ["y"],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, expected_out_shape)]
        )

    def test_qlinearmatmul_allow_unknown(self) -> None:
        self._make_qlinearmatmul_test_allow_unknown((None,), (None,), ())
        self._make_qlinearmatmul_test_allow_unknown((3,), (None,), ())
        self._make_qlinearmatmul_test_allow_unknown((2,), (2, "a"), ("a",))
        self._make_qlinearmatmul_test_allow_unknown((4, 2), (2, "a"), (4, "a"))
        self._make_qlinearmatmul_test_allow_unknown((4, None), (2, "a"), (4, "a"))
        self._make_qlinearmatmul_test_allow_unknown((4, None), (None, "a"), (4, "a"))
        self._make_qlinearmatmul_test_allow_unknown((1, 4, 2), ("a", 2, 5), ("a", 4, 5))
        self._make_qlinearmatmul_test_allow_unknown(
            (1, 3, 4, 2), ("a", 2, 5), (1, 3, 4, 5)
        )
        self._make_qlinearmatmul_test_allow_unknown(None, ("a", 2, 5), None)
        self._make_qlinearmatmul_test_allow_unknown(None, None, None)

    def _make_matmulinteger_test(
        self, shape1: Sequence[int], shape2: Sequence[int]
    ) -> None:
        expected_out_shape = np.matmul(
            np.arange(np.prod(shape1)).reshape(shape1),
            np.arange(np.prod(shape2)).reshape(shape2),
        ).shape
        graph = self._make_graph(
            [
                ("A", TensorProto.UINT8, shape1),
                ("B", TensorProto.UINT8, shape2),
                ("a_zero_point", TensorProto.UINT8, ()),
                ("b_zero_point", TensorProto.UINT8, ()),
            ],
            [
                make_node(
                    "MatMulInteger", ["A", "B", "a_zero_point", "b_zero_point"], ["Y"]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.INT32, expected_out_shape)]
        )

    def test_matmulinteger(self) -> None:
        self._make_matmulinteger_test((2,), (2,))
        self._make_matmulinteger_test((1, 2), (2, 3))
        self._make_matmulinteger_test((2,), (2, 3))
        self._make_matmulinteger_test((4, 2), (2,))
        self._make_matmulinteger_test((5, 1, 4, 2), (1, 3, 2, 3))
        self._make_matmulinteger_test((4, 2), (3, 2, 3))

    @parameterized.expand(
        [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16]
    )
    def test_quantizelinear(self, elem_type) -> None:
        graph = self._make_graph(
            [
                ("x", elem_type, (30, 4, 5)),
                ("y_scale", elem_type, ()),
                ("y_zero_point", TensorProto.UINT8, ()),
            ],
            [make_node("QuantizeLinear", ["x", "y_scale", "y_zero_point"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (30, 4, 5))]
        )

    def test_quantizelinear_default_zp(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (30, 4, 5)), ("y_scale", TensorProto.FLOAT, ())],
            [make_node("QuantizeLinear", ["x", "y_scale"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (30, 4, 5))]
        )

    def test_quantizelinear_optional_input(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (30, 4, 5)), ("y_scale", TensorProto.FLOAT, ())],
            [make_node("QuantizeLinear", ["x", "y_scale", ""], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (30, 4, 5))]
        )

    def test_quantizelinear_output_dtype(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("y_scale", TensorProto.FLOAT, ())],
            [
                make_node(
                    "QuantizeLinear",
                    ["x", "y_scale"],
                    ["y"],
                    output_dtype=TensorProto.UINT4,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT4, (3, 4, 5))]
        )

    def test_quantizelinear_zp_output_dtype(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5)),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT16, ()),
            ],
            [
                make_node(
                    "QuantizeLinear",
                    ["x", "y_scale", "y_zero_point"],
                    ["y"],
                    output_dtype=TensorProto.UINT16,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT16, (3, 4, 5))]
        )

    def test_quantizelinear_zp_output_dtype_conflicted(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5)),
                ("y_scale", TensorProto.FLOAT, ()),
                ("y_zero_point", TensorProto.UINT16, ()),
            ],
            [
                make_node(
                    "QuantizeLinear",
                    ["x", "y_scale", "y_zero_point"],
                    ["y"],
                    output_dtype=TensorProto.INT4,
                )
            ],
            [],
        )

        self.assertRaises(
            onnx.shape_inference.InferenceError,
            self._inferred,
            graph,
        )

    @unittest.skip(
        "Issue #5960"
    )  # FIXME(#5960) propagateElemTypeFromAttributeToOutput does not validate against output type constraints
    def test_quantizelinear_invalid_output_dtype(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("y_scale", TensorProto.FLOAT, ())],
            [
                make_node(
                    "QuantizeLinear",
                    ["x", "y_scale"],
                    ["y"],
                    output_dtype=TensorProto.FLOAT16,
                )
            ],
            [],
        )

        self.assertRaises(
            onnx.shape_inference.InferenceError,
            self._inferred,
            graph,
        )

    @parameterized.expand(
        [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16]
    )
    def test_dequantizelinear(self, elem_type) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.UINT8, (30, 4, 5)),
                ("x_scale", elem_type, ()),
                ("x_zero_point", TensorProto.UINT8, ()),
            ],
            [make_node("DequantizeLinear", ["x", "x_scale", "x_zero_point"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", elem_type, (30, 4, 5))]
        )

    def test_dynamicquantizelinear(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (30, 4, 5))],
            [
                make_node(
                    "DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"]
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.UINT8, (30, 4, 5)),
                make_tensor_value_info("y_scale", TensorProto.FLOAT, ()),
                make_tensor_value_info("y_zero_point", TensorProto.UINT8, ()),
            ],
        )

    def test_reversesequence(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (4, 5, 6)),
                ("sequence_lens", TensorProto.INT64, (5,)),
            ],
            [make_node("ReverseSequence", ["x", "sequence_lens"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (4, 5, 6))]
        )

    def test_unique_without_axis(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 4, 2))],
            [make_node("Unique", ["X"], ["Y", "indices", "inverse_indices", "counts"])],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("Y", TensorProto.FLOAT, (None,)),  # type: ignore
                make_tensor_value_info("indices", TensorProto.INT64, (None,)),  # type: ignore
                make_tensor_value_info("inverse_indices", TensorProto.INT64, (None,)),  # type: ignore
                make_tensor_value_info("counts", TensorProto.INT64, (None,)),
            ],
        )  # type: ignore

    def test_unique_with_axis(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (2, 4, 2))],
            [
                make_node(
                    "Unique",
                    ["X"],
                    ["Y", "indices", "inverse_indices", "counts"],
                    axis=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("Y", TensorProto.FLOAT, (2, None, 2)),  # type: ignore
                make_tensor_value_info("indices", TensorProto.INT64, (None,)),  # type: ignore
                make_tensor_value_info("inverse_indices", TensorProto.INT64, (None,)),  # type: ignore
                make_tensor_value_info("counts", TensorProto.INT64, (None,)),
            ],
        )  # type: ignore

    def test_det(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (3, 3))], [make_node("Det", ["X"], ["Y"])], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, ())]
        )

        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (4, 5, 6, 7, 7))],
            [make_node("Det", ["X"], ["Y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (4, 5, 6))]
        )

    def test_tile(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6)), ("repeats", TensorProto.INT64, (3,))],
            [make_node("Tile", ["x", "repeats"], ["y"])],
            [],
            initializer=[make_tensor("repeats", TensorProto.INT64, (3,), (1, 2, 3))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (4, 10, 18))]
        )

    def test_tile_raw_input_data(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6)), ("repeats", TensorProto.INT64, (3,))],
            [make_node("Tile", ["x", "repeats"], ["y"])],
            [],
            initializer=[
                make_tensor(
                    "repeats",
                    TensorProto.INT64,
                    (3,),
                    vals=np.array([1, 2, 3], dtype="<i8").tobytes(),
                    raw=True,
                )
            ],
        )  # Feed raw bytes (force little endian ordering like onnx standard) for test purpose
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (4, 10, 18))]
        )

    def test_tile_rank_inference(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6)), ("repeats", TensorProto.INT64, (3,))],
            [make_node("Tile", ["x", "repeats"], ["y"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (None, None, None))])  # type: ignore

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_linearclassifier_1D_input(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (5,))],
            [
                make_node(
                    "LinearClassifier",
                    ["x"],
                    ["y", "z"],
                    domain=ONNX_ML_DOMAIN,
                    coefficients=[0.0008, -0.0008],
                    intercepts=[2.0, 2.0],
                    classlabels_ints=[1, 2],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.INT64, (1,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (1, 2)),
            ],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_linearclassifier_2D_input(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5))],
            [
                make_node(
                    "LinearClassifier",
                    ["x"],
                    ["y", "z"],
                    domain=ONNX_ML_DOMAIN,
                    coefficients=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    intercepts=[2.0, 2.0, 3.0],
                    classlabels_ints=[1, 2, 3],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.INT64, (4,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (4, 3)),
            ],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    def test_roialign_symbolic(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", "H", "W")),
                ("rois", TensorProto.FLOAT, ("num_rois", 4)),
                ("batch_indices", TensorProto.INT64, ("num_rois",)),
            ],
            [
                make_node(
                    "RoiAlign",
                    ["x", "rois", "batch_indices"],
                    ["y"],
                    output_height=10,
                    output_width=5,
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, ("num_rois", "C", 10, 5))])  # type: ignore

    def test_roialign_symbolic_defaults(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", "H", "W")),
                ("rois", TensorProto.FLOAT, ("num_rois", 4)),
                ("batch_indices", TensorProto.INT64, ("num_rois",)),
            ],
            [make_node("RoiAlign", ["x", "rois", "batch_indices"], ["y"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, ("num_rois", "C", 1, 1))])  # type: ignore

    def test_roialign_num_rois(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", "H", "W")),
                ("rois", TensorProto.FLOAT, ("num_rois", 4)),
                ("batch_indices", TensorProto.INT64, (15,)),
            ],
            [make_node("RoiAlign", ["x", "rois", "batch_indices"], ["y"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (15, "C", 1, 1))])  # type: ignore

    @parameterized.expand(
        all_versions_for("LabelEncoder") if ONNX_ML else [], skip_on_empty=True
    )
    def test_label_encoder_string_int64(self, _, version) -> None:
        self.skipIf(
            version < 2, "keys_* attributes were introduced in ai.onnx.ml opset 2"
        )
        string_list = ["A", "m", "y"]
        float_list = [94.17, 36.00, -99.0]
        int64_list = [12, 28, 86]
        graph = self._make_graph(
            [("x", TensorProto.STRING, (6, 1))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_strings=string_list,
                    values_int64s=int64_list,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (6, 1))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

        graph = self._make_graph(
            [("x", TensorProto.INT64, (2, 3))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_int64s=int64_list,
                    values_strings=string_list,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.STRING, (2, 3))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2,))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_floats=float_list,
                    values_int64s=int64_list,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (2,))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

        graph = self._make_graph(
            [("x", TensorProto.INT64, (8,))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_int64s=int64_list,
                    values_floats=float_list,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (8,))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ())],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_floats=float_list,
                    values_strings=string_list,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.STRING, ())],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

        graph = self._make_graph(
            [("x", TensorProto.STRING, (1, 2))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_strings=string_list,
                    values_floats=float_list,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (1, 2))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @parameterized.expand(
        all_versions_for("LabelEncoder") if ONNX_ML else [], skip_on_empty=True
    )
    def test_label_encoder_tensor_attributes(self, _, version) -> None:
        self.skipIf(
            version < 4, "tensor attributes were introduced in ai.onnx.ml opset 4"
        )
        key_tensor = make_tensor(
            "keys_tensor", TensorProto.STRING, [4], ["a", "b", "cc", "ddd"]
        )
        values_tensor = make_tensor(
            "values_tensor", TensorProto.INT64, [4], [1, 2, 3, 4]
        )
        graph = self._make_graph(
            [("x", TensorProto.STRING, ("M", None, 3, 12))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_tensor=key_tensor,
                    values_tensor=values_tensor,
                    default_tensor=make_tensor(
                        "default_tensor", TensorProto.INT64, [1], [0]
                    ),
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, ("M", None, 3, 12))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, version),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @parameterized.expand(
        all_versions_for("LabelEncoder") if ONNX_ML else [], skip_on_empty=True
    )
    def test_label_encoder_tensor_attributes_invalid_configurations(
        self, _, version
    ) -> None:
        self.skipIf(version < 4, "tensor attributes introduced in ai.onnx.ml opset 4")
        key_tensor = make_tensor(
            "keys_tensor", TensorProto.STRING, [4], ["a", "b", "cc", "ddd"]
        )
        values_tensor = make_tensor(
            "values_tensor", TensorProto.INT64, [4], [1, 2, 3, 4]
        )

        opset_imports = [
            make_opsetid(ONNX_ML_DOMAIN, version),
            make_opsetid(ONNX_DOMAIN, 11),
        ]

        # default_tensor should be INT64, same type as values_tensor
        graph = self._make_graph(
            [("x", TensorProto.STRING, ("M", None, 3, 12))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_tensor=key_tensor,
                    values_tensor=values_tensor,
                    default_tensor=make_tensor(
                        "default_tensor", TensorProto.STRING, [1], [0]
                    ),
                )
            ],
            [],
        )

        self.assertRaises(
            onnx.shape_inference.InferenceError,
            self._inferred,
            graph,
            opset_imports=opset_imports,
        )

        # default_tensor should be a singleton of shape (1,)
        graph = self._make_graph(
            [("x", TensorProto.STRING, ("M", None, 3, 12))],
            [
                make_node(
                    "LabelEncoder",
                    ["x"],
                    ["y"],
                    domain=ONNX_ML_DOMAIN,
                    keys_tensor=key_tensor,
                    values_strings=["a", "b", "cc", "ddd"],
                    default_tensor=make_tensor(
                        "default_tensor", TensorProto.STRING, [1, 2], [0, 0]
                    ),
                )
            ],
            [],
        )

        self.assertRaises(
            onnx.shape_inference.InferenceError,
            self._inferred,
            graph,
            opset_imports=opset_imports,
        )

    def make_sparse(
        self,
        shape: Sequence[int],
        values: Sequence[int],
        indices_shape: Sequence[int],
        indices: Sequence[int],
    ) -> SparseTensorProto:
        sparse = SparseTensorProto()
        sparse.dims.extend(shape)
        nnz = len(values)
        sparse.values.CopyFrom(
            helper.make_tensor("spval", TensorProto.INT64, (nnz,), values)
        )
        sparse.indices.CopyFrom(
            helper.make_tensor("spind", TensorProto.INT64, indices_shape, indices)
        )
        return sparse

    def test_constant_sparse(self) -> None:
        y_shape = [100]
        y_value = self.make_sparse(y_shape, [13, 17, 19], [3], [9, 27, 81])
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], sparse_value=y_value)], []
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.INT64, y_shape)])  # type: ignore

    def test_constant_value_int(self) -> None:
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], value_int=42)], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT64, [])]
        )

    def test_constant_value_ints(self) -> None:
        value_ints = [1, 2, 3]
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], value_ints=value_ints)], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.INT64, [len(value_ints)])]
        )

    def test_constant_value_float(self) -> None:
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], value_float=1.42)], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, [])]
        )

    def test_constant_value_floats(self) -> None:
        value_floats = [1.0, 1.1, 1.2]
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], value_floats=value_floats)], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, [len(value_floats)])]
        )

    def test_constant_value_string(self) -> None:
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], value_string="String value")], []
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.STRING, [])]
        )

    def test_constant_value_strings(self) -> None:
        value_strings = ["o", "n", "n", "x"]
        graph = self._make_graph(
            [], [make_node("Constant", [], ["y"], value_strings=value_strings)], []
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.STRING, [len(value_strings)])],
        )

    def test_range(self) -> None:
        graph = self._make_graph(
            [
                ("start", TensorProto.FLOAT, ()),
                ("limit", TensorProto.FLOAT, ()),
                ("delta", TensorProto.FLOAT, ()),
            ],
            [make_node("Range", ["start", "limit", "delta"], ["output"])],
            [],
            initializer=[
                make_tensor("start", TensorProto.FLOAT, (), (1,)),
                make_tensor("limit", TensorProto.FLOAT, (), (5,)),
                make_tensor("delta", TensorProto.FLOAT, (), (2,)),
            ],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("output", TensorProto.FLOAT, (2,))]
        )

    def test_range_rank_inference(self) -> None:
        graph = self._make_graph(
            [
                ("start", TensorProto.INT32, ()),
                ("limit", TensorProto.INT32, ()),
                ("delta", TensorProto.INT32, ()),
            ],
            [make_node("Range", ["start", "limit", "delta"], ["output"])],
            [],
            initializer=[
                make_tensor("start", TensorProto.INT32, (), (1,)),
                make_tensor("limit", TensorProto.INT32, (), (5,)),
            ],
        )  # Missing 'delta' initializer
        self._assert_inferred(graph, [make_tensor_value_info("output", TensorProto.INT32, (None,))])  # type: ignore

    def test_gathernd(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (4, 5, 6)), ("indices", TensorProto.INT64, (2,))],
            [make_node("GatherND", ["x", "indices"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (6,))]
        )

    def test_gathernd_batchdim_1(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (2, 2, 2)),
                ("indices", TensorProto.INT64, (2, 1)),
            ],
            [make_node("GatherND", ["x", "indices"], ["y"], batch_dims=1)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2, 2))]
        )

    def test_cumsum(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)), ("axis", TensorProto.FLOAT, (1,))],
            [make_node("CumSum", ["x", "axis"], "z")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3))]
        )

    def test_nonmaxsuppression(self) -> None:
        graph = self._make_graph(
            [
                ("boxes", TensorProto.FLOAT, (1, 3, 4)),
                ("scores", TensorProto.FLOAT, (1, 5, 3)),
            ],
            [make_node("NonMaxSuppression", ["boxes", "scores"], ["y"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.INT64, (None, 3))])  # type: ignore

    def test_sequence_empty(self) -> None:
        graph = self._make_graph([], [make_node("SequenceEmpty", [], ["output"])], [])
        self._assert_inferred(graph, [make_tensor_sequence_value_info("output", TensorProto.FLOAT, None)])  # type: ignore

    def test_sequence_construct(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
            ],
            [
                make_node(
                    "SequenceConstruct",
                    ["input1", "input2", "input3"],
                    ["output_sequence"],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, 3, 4)
                )
            ],
        )  # type: ignore

    def test_sequence_construct_one_input(self) -> None:
        graph = self._make_graph(
            [("input1", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("SequenceConstruct", ["input1"], ["output_sequence"])],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, 3, 4)
                )
            ],
        )  # type: ignore

    def test_sequence_construct_diff_rank(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3)),
                ("input3", TensorProto.FLOAT, (2, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct",
                    ["input1", "input2", "input3"],
                    ["output_sequence"],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, None
                )
            ],
        )  # type: ignore

    def test_sequence_construct_diff_dim_size(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 5)),
                ("input3", TensorProto.FLOAT, (2, 3, 6)),
            ],
            [
                make_node(
                    "SequenceConstruct",
                    ["input1", "input2", "input3"],
                    ["output_sequence"],
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, 3, None)
                )
            ],
        )  # type: ignore

    def test_sequence_insert(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
                ("input4", TensorProto.FLOAT, (2, 3, 4)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceInsert", ["in_sequence", "input4"], ["output_sequence"]
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, 4)
                ),
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, 3, 4)
                ),
            ],
        )  # type: ignore

    def test_sequence_insert_diff_rank(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
                ("input4", TensorProto.FLOAT, (2, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceInsert", ["in_sequence", "input4"], ["output_sequence"]
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, 4)
                ),
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, None
                ),
            ],
        )  # type: ignore

    def test_sequence_insert_diff_shape(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 5, 4)),
                ("input4", TensorProto.FLOAT, (2, 5, 2)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceInsert", ["in_sequence", "input4"], ["output_sequence"]
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, None, 4)),  # type: ignore
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, None, None)
                ),
            ],
        )  # type: ignore

    def test_sequence_at(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
                ("ind", TensorProto.INT64, ()),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceAt", ["in_sequence", "ind"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, 4)
                ),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 4)),
            ],
        )  # type: ignore

    def test_sequence_at_unknown_shape(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
                ("ind", TensorProto.INT64, ()),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceAt", ["in_sequence", "ind"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, None),
                make_tensor_value_info("output", TensorProto.FLOAT, None),
            ],
        )  # type: ignore

    def test_sequence_at_unknown_dim_size(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 5)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
                ("ind", TensorProto.INT64, ()),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceAt", ["in_sequence", "ind"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, 3, None)),  # type: ignore
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, None)),
            ],
        )  # type: ignore

    def test_sequence_erase(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, 4)),
                ("input2", TensorProto.FLOAT, (2, 3, 4)),
                ("input3", TensorProto.FLOAT, (2, 3, 4)),
                ("ind", TensorProto.INT64, ()),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceErase", ["in_sequence", "ind"], ["output_sequence"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, 4)
                ),
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, 3, 4)
                ),
            ],
        )  # type: ignore

    def test_sequence_erase_diff_dim_size(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 3, "x")),
                ("input3", TensorProto.FLOAT, (2, 5, "x")),
                ("ind", TensorProto.INT64, ()),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceErase", ["in_sequence", "ind"], ["output_sequence"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, None, "x")),  # type: ignore
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, None, "x")
                ),
            ],
        )  # type: ignore

    def test_sequence_length(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 3, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceLength", ["in_sequence"], ["len"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, "x")
                ),
                make_tensor_value_info("len", TensorProto.INT64, ()),
            ],
        )  # type: ignore

    def test_split_to_sequence(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4)), ("split", TensorProto.INT32, (2,))],
            [make_node("SplitToSequence", ["input", "split"], ["output_sequence"])],
            [],
            initializer=[make_tensor("split", TensorProto.INT32, (2,), (3, 3))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (3, 4)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_scalar(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4)), ("split", TensorProto.INT32, ())],
            [make_node("SplitToSequence", ["input", "split"], ["output_sequence"])],
            [],
            initializer=[make_tensor("split", TensorProto.INT32, (), (2,))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (2, 4)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_keepdims(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4))],
            [make_node("SplitToSequence", ["input"], ["output_sequence"], keepdims=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (1, 4)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_not_keepdims(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4))],
            [make_node("SplitToSequence", ["input"], ["output_sequence"], keepdims=0)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (4,)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_ignore_keepdims(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4)), ("split", TensorProto.INT32, (2,))],
            [
                make_node(
                    "SplitToSequence",
                    ["input", "split"],
                    ["output_sequence"],
                    keepdims=0,
                )
            ],
            [],
            initializer=[make_tensor("split", TensorProto.INT32, (2,), (3, 3))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (3, 4)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_axis(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4))],
            [make_node("SplitToSequence", ["input"], ["output_sequence"], axis=1)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (6, 1)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_neg_axis(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4))],
            [make_node("SplitToSequence", ["input"], ["output_sequence"], axis=-2)],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (1, 4)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_split_sizes(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4)), ("split", TensorProto.INT32, (3,))],
            [make_node("SplitToSequence", ["input", "split"], ["output_sequence"])],
            [],
            initializer=[make_tensor("split", TensorProto.INT32, (3,), (2, 1, 3))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (None, 4)
                )
            ],
        )  # type: ignore

    def test_split_to_sequence_non_divisible(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (6, 4)), ("split", TensorProto.INT32, ())],
            [make_node("SplitToSequence", ["input", "split"], ["output_sequence"])],
            [],
            initializer=[make_tensor("split", TensorProto.INT32, (), (4,))],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "output_sequence", TensorProto.FLOAT, (None, 4)
                )
            ],
        )  # type: ignore

    def test_concat_from_sequence(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 3, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("ConcatFromSequence", ["in_sequence"], ["out"], axis=0),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, "x")
                ),
                make_tensor_value_info("out", TensorProto.FLOAT, (None, 3, "x")),
            ],
        )  # type: ignore

    def test_concat_from_sequence_unknown_shape(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 3)),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("ConcatFromSequence", ["in_sequence"], ["out"], axis=0),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, None),
                make_tensor_value_info("out", TensorProto.FLOAT, None),
            ],
        )  # type: ignore

    def test_concat_from_sequence_unknown_dim_size(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 4, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("ConcatFromSequence", ["in_sequence"], ["out"], axis=0),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, None, "x")),  # type: ignore
                make_tensor_value_info("out", TensorProto.FLOAT, (None, None, "x")),
            ],
        )  # type: ignore

    def test_concat_from_sequence_axis(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 4, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("ConcatFromSequence", ["in_sequence"], ["out"], axis=2),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, None, "x")),  # type: ignore
                make_tensor_value_info("out", TensorProto.FLOAT, (2, None, None)),
            ],
        )  # type: ignore

    def test_concat_from_sequence_neg_axis(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 4, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("ConcatFromSequence", ["in_sequence"], ["out"], axis=-3),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info("in_sequence", TensorProto.FLOAT, (2, None, "x")),  # type: ignore
                make_tensor_value_info("out", TensorProto.FLOAT, (None, None, "x")),
            ],
        )  # type: ignore

    def test_concat_from_sequence_new_axis(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 3, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "ConcatFromSequence", ["in_sequence"], ["out"], axis=2, new_axis=1
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, "x")
                ),
                make_tensor_value_info("out", TensorProto.FLOAT, (2, 3, None, "x")),
            ],
        )  # type: ignore

    def test_concat_from_sequence_neg_new_axis(self) -> None:
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (2, 3, "x")),
                ("input2", TensorProto.FLOAT, (2, 3, "x")),
                ("input3", TensorProto.FLOAT, (2, 3, "x")),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "ConcatFromSequence", ["in_sequence"], ["out"], axis=-1, new_axis=1
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (2, 3, "x")
                ),
                make_tensor_value_info("out", TensorProto.FLOAT, (2, 3, "x", None)),
            ],
        )  # type: ignore

    def test_adagrad(self) -> None:
        graph = self._make_graph(
            [
                ("R", TensorProto.FLOAT, ()),  # scalar's shape is ()
                ("T", TensorProto.INT64, ()),  # scalar's shape is ()
                ("X", TensorProto.FLOAT, (1, 2)),
                ("G", TensorProto.FLOAT, (1, 2)),
                ("H", TensorProto.FLOAT, (1, 2)),
            ],
            [
                make_node(
                    "Adagrad",
                    ["R", "T", "X", "G", "H"],
                    ["X_new", "H_new"],
                    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("X_new", TensorProto.FLOAT, (1, 2)),
                make_tensor_value_info("H_new", TensorProto.FLOAT, (1, 2)),
            ],
            opset_imports=[
                helper.make_opsetid(ONNX_DOMAIN, 12),
                helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
            ],
        )

    def test_adagrad_multiple(self) -> None:
        graph = self._make_graph(
            [
                ("R", TensorProto.FLOAT, ()),  # scalar's shape is ()
                ("T", TensorProto.INT64, ()),  # scalar's shape is ()
                ("X1", TensorProto.FLOAT, (1, 2)),
                ("X2", TensorProto.FLOAT, (3, 4)),
                ("G1", TensorProto.FLOAT, (1, 2)),
                ("G2", TensorProto.FLOAT, (3, 4)),
                ("H1", TensorProto.FLOAT, (1, 2)),
                ("H2", TensorProto.FLOAT, (3, 4)),
            ],
            [
                make_node(
                    "Adagrad",
                    ["R", "T", "X1", "X2", "G1", "G2", "H1", "H2"],
                    ["X1_new", "X2_new", "H1_new", "H2_new"],
                    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("X1_new", TensorProto.FLOAT, (1, 2)),
                make_tensor_value_info("X2_new", TensorProto.FLOAT, (3, 4)),
                make_tensor_value_info("H1_new", TensorProto.FLOAT, (1, 2)),
                make_tensor_value_info("H2_new", TensorProto.FLOAT, (3, 4)),
            ],
            opset_imports=[
                helper.make_opsetid(ONNX_DOMAIN, 12),
                helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
            ],
        )

    def test_momentum(self) -> None:
        graph = self._make_graph(
            [
                ("R", TensorProto.FLOAT, ()),  # scalar's shape is ()
                ("T", TensorProto.INT64, ()),  # scalar's shape is ()
                ("X", TensorProto.FLOAT, (1, 2)),
                ("G", TensorProto.FLOAT, (1, 2)),
                ("V", TensorProto.FLOAT, (1, 2)),
            ],
            [
                make_node(
                    "Momentum",
                    ["R", "T", "X", "G", "V"],
                    ["X_new", "V_new"],
                    alpha=0.9,
                    beta=1.0,
                    norm_coefficient=0.02,
                    mode="standard",
                    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("X_new", TensorProto.FLOAT, (1, 2)),
                make_tensor_value_info("V_new", TensorProto.FLOAT, (1, 2)),
            ],
            opset_imports=[
                helper.make_opsetid(ONNX_DOMAIN, 12),
                helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
            ],
        )

    def test_momentum_multiple(self) -> None:
        graph = self._make_graph(
            [
                ("R", TensorProto.FLOAT, ()),  # scalar's shape is ()
                ("T", TensorProto.INT64, ()),  # scalar's shape is ()
                ("X1", TensorProto.FLOAT, (1, 2)),
                ("X2", TensorProto.FLOAT, (3, 4)),
                ("G1", TensorProto.FLOAT, (1, 2)),
                ("G2", TensorProto.FLOAT, (3, 4)),
                ("V1", TensorProto.FLOAT, (1, 2)),
                ("V2", TensorProto.FLOAT, (3, 4)),
            ],
            [
                make_node(
                    "Momentum",
                    ["R", "T", "X1", "X2", "G1", "G2", "V1", "V2"],
                    ["X1_new", "X2_new", "V1_new", "V2_new"],
                    alpha=0.9,
                    beta=1.0,
                    norm_coefficient=0.02,
                    mode="nesterov",
                    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
                )
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("X1_new", TensorProto.FLOAT, (1, 2)),
                make_tensor_value_info("X2_new", TensorProto.FLOAT, (3, 4)),
                make_tensor_value_info("V1_new", TensorProto.FLOAT, (1, 2)),
                make_tensor_value_info("V2_new", TensorProto.FLOAT, (3, 4)),
            ],
            opset_imports=[
                helper.make_opsetid(ONNX_DOMAIN, 12),
                helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
            ],
        )

    def test_adam(self) -> None:
        graph = self._make_graph(
            [
                ("R", TensorProto.FLOAT, ()),  # scalar's shape is ()
                ("T", TensorProto.INT64, ()),  # scalar's shape is ()
                ("X", TensorProto.FLOAT, (1, 2)),
                ("G", TensorProto.FLOAT, (1, 2)),
                ("V", TensorProto.FLOAT, (1, 2)),
                ("H", TensorProto.FLOAT, (1, 2)),
            ],
            [
                make_node(
                    "Adam",
                    ["R", "T", "X", "G", "V", "H"],
                    ["X_new", "V_new", "H_new"],
                    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
                    alpha=0.9,
                    beta=1.0,
                    norm_coefficient=0.02,
                )
            ],
            [],
        )

        infos = [
            make_tensor_value_info("X_new", TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("V_new", TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("H_new", TensorProto.FLOAT, (1, 2)),
        ]

        self._assert_inferred(
            graph,
            infos,
            opset_imports=[
                make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 12),
            ],
        )

    def test_adam_multiple(self) -> None:
        graph = self._make_graph(
            [
                ("R", TensorProto.FLOAT, ()),  # scalar's shape is ()
                ("T", TensorProto.INT64, ()),  # scalar's shape is ()
                ("X1", TensorProto.FLOAT, (1, 2)),
                ("X2", TensorProto.FLOAT, (3, 4)),
                ("G1", TensorProto.FLOAT, (1, 2)),
                ("G2", TensorProto.FLOAT, (3, 4)),
                ("V1", TensorProto.FLOAT, (1, 2)),
                ("V2", TensorProto.FLOAT, (3, 4)),
                ("H1", TensorProto.FLOAT, (1, 2)),
                ("H2", TensorProto.FLOAT, (3, 4)),
            ],
            [
                make_node(
                    "Adam",
                    ["R", "T", "X1", "X2", "G1", "G2", "V1", "V2", "H1", "H2"],
                    ["X1_new", "X2_new", "V1_new", "V2_new", "H1_new", "H2_new"],
                    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
                    alpha=0.9,
                    beta=1.0,
                    norm_coefficient=0.02,
                )
            ],
            [],
        )

        infos = [
            make_tensor_value_info("X1_new", TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("X2_new", TensorProto.FLOAT, (3, 4)),
            make_tensor_value_info("V1_new", TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("V2_new", TensorProto.FLOAT, (3, 4)),
            make_tensor_value_info("H1_new", TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("H2_new", TensorProto.FLOAT, (3, 4)),
        ]

        self._assert_inferred(
            graph,
            infos,
            opset_imports=[
                make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 12),
            ],
        )

    def test_pad_opset10(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (1, None, 2))],
            [make_node("Pad", "x", "y", pads=[1, 3, 1, 1, 0, 1])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (3, None, 4))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)],
        )  # type: ignore

    def test_constant_pad_2d_opset10(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4, 4))],
            [
                make_node(
                    "Pad",
                    "x",
                    "y",
                    pads=[0, 0, 3, 1, 0, 0, 4, 2],
                    mode="constant",
                    value=2.0,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (2, 3, 11, 7))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 10)],
        )

    def test_pad(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (1, None, 2)), ("pads", TensorProto.INT64, (6,))],
            [make_node("Pad", ["x", "pads"], "y")],
            [],
            initializer=[
                make_tensor(
                    "pads",
                    TensorProto.INT64,
                    (6,),
                    (
                        1,
                        3,
                        1,
                        1,
                        0,
                        1,
                    ),
                )
            ],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, None, 4))])  # type: ignore

    def test_gatherelements_basic(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (6,)), ("indices", TensorProto.INT64, (2,))],
            [make_node("GatherElements", ["x", "indices"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (2,))]
        )

    def test_gatherelements_indices_missing_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (6,)),
                ("indices", TensorProto.INT64, None),
            ],  # type: ignore
            [make_node("GatherElements", ["x", "indices"], ["y"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, None)])  # type: ignore

    def test_einsum_transpose(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4))],
            [make_node("Einsum", ["x"], ["y"], equation="ij->ji")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (4, 3))])  # type: ignore

    def test_einsum_dot(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (1,)), ("y", TensorProto.FLOAT, (1,))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="i,i->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_scalar(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ()), ("y", TensorProto.FLOAT, ())],
            [make_node("Einsum", ["x", "y"], ["z"], equation=",->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_outer_prod(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 5)), ("y", TensorProto.FLOAT, (7, 9))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,ab->ijab")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 5, 7, 9))])  # type: ignore

    def test_einsum_sum_along_dim(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4))],
            [make_node("Einsum", ["x"], ["y"], equation="i j->i ")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_ellipsis(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 4))],
            [make_node("Einsum", ["x"], ["y"], equation="... ii ->... i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 4))])  # type: ignore

    def test_einsum_ellipsis_2(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4)), ("y", TensorProto.FLOAT, (2, 4, 5))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="...ij,...jk->...ik")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 5))]
        )  # type: ignore

    def test_einsum_ellipsis_3(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 4)), ("y", TensorProto.FLOAT, (2, 4, 5))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="...ij,...jk")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 5))]
        )  # type: ignore

    def test_einsum_ellipsis_broadcast(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (1, 3, 4)), ("y", TensorProto.FLOAT, (32, 4, 5))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="...ij,...jk->...ik")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (32, 3, 5))]
        )  # type: ignore

    def test_einsum_contraction(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (5, 6, 7, 8)),
                ("y", TensorProto.FLOAT, (8, 9, 10)),
            ],
            [make_node("Einsum", ["x", "y"], ["z"], equation="abcd,dfg->abcfg")],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("z", TensorProto.FLOAT, (5, 6, 7, 9, 10))],
        )  # type: ignore

    def test_einsum_contraction_2(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("y", TensorProto.FLOAT, (3, 5))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ijk,ik->jk")],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("z", TensorProto.FLOAT, (4, 5))]
        )  # type: ignore

    def test_einsum_batch_matmul(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (5, 2, 3)), ("y", TensorProto.FLOAT, (5, 3, 4))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="bij , b jk-> bik")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (5, 2, 4))])  # type: ignore

    def test_einsum_left_hand_eqn(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)), ("y", TensorProto.FLOAT, (3, 4))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,kl")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 3, 4))])  # type: ignore

    def test_einsum_incorrect_num_inputs(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (2, 3)),
                ("y", TensorProto.FLOAT, (2, 3)),
                ("z", TensorProto.FLOAT, (2, 3)),
            ],
            [make_node("Einsum", ["x", "y"], ["z"], equation="i,...j, k, l-> i")],
            [],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    def test_einsum_view_A1(self) -> None:  # returns a view of A1
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,))],
            [make_node("Einsum", ["x"], ["y"], equation="i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_sum_A1(self) -> None:  # sums the values of A1
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,))],
            [make_node("Einsum", ["x"], ["y"], equation="i->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_element_wise_multiplication_A1_B1(
        self,
    ) -> None:  # element-wise multiplication of A1 and B1
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,)), ("y", TensorProto.FLOAT, (3,))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="i,i->i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_inner_product_A1_B1(self) -> None:  # inner product of A1 and B1
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,)), ("y", TensorProto.FLOAT, (3,))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="i,i->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_outer_product_A1_B1(self) -> None:  # outer product of A1 and B1
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,)), ("y", TensorProto.FLOAT, (3,))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="i,j->ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_view_A2(self) -> None:  # returns a view of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ij->ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_view_A2_2(self) -> None:  # returns a view of A2, another case
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_transpose_A2(self) -> None:  # view transpose of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ji")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_transpose_A2_to_ij(self) -> None:  # view transpose of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ji->ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_diag_A2(self) -> None:  # view main diagonal of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ii->i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_trace_A2(self) -> None:  # sums main diagonal of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ii->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_sum_A2(self) -> None:  # sums the values of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ij->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, ())])  # type: ignore

    def test_einsum_sum_columns_A2(
        self,
    ) -> None:  # sum down the columns of A2 (across rows)
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ij->j")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_sum_rows_A2(self) -> None:  # sum horizontally along the rows of A2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x"], ["y"], equation="ij->i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_element_wise_multiplication_A2_B2(
        self,
    ) -> None:  # element-wise multiplication of A2 and B2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,ij->ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_element_wise_multiplication_A2_B2_transpose(
        self,
    ) -> None:  # element-wise multiplication of A2 and B2.T
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,ji->ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_matrix_multiplication_A2_B2(
        self,
    ) -> None:  # matrix multiplication of A2 and B2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,jk")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_matrix_multiplication_A2_B2_to_ik(
        self,
    ) -> None:  # matrix multiplication of A2 and B2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,jk->ik")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_matrix_multiplication_A3_B3(
        self,
    ) -> None:  # matrix multiplication of A3 and B3 (a stack of 2D matrices)
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 3)), ("y", TensorProto.FLOAT, (2, 3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="bij,bjk->bik")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 3))])  # type: ignore

    def test_einsum_matrix_multiplication_A3_B3_transpose(
        self,
    ) -> None:  # matrix multiplication of A3 and B3 (a stack of 2D matrices)
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3, 3)), ("y", TensorProto.FLOAT, (2, 3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="bij,bkj->bik")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 3, 3))])  # type: ignore

    def test_einsum_inner_product_A2_B2(self) -> None:  # inner product of A2 and B2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,kj->ik")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_row_multiplication_A2_B2(
        self,
    ) -> None:  # each row of A2 multiplied by B2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,kj->ikj")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3, 3))])  # type: ignore

    def test_einsum_value_multiplication_A2_B2(
        self,
    ) -> None:  # each value of A2 multiplied by B2
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,kl->ijkl")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3, 3, 3))])  # type: ignore

    def test_einsum_scalar_times_array(self) -> None:  # Scalar times array
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ()), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation=",ij->ij")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3, 3))])  # type: ignore

    def test_einsum_matrix_vector_A2_B1(self) -> None:  # Matrix and vector.
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3,))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ij,j->i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_diag_multiplication_A2_B2(
        self,
    ) -> None:  # diagonals multiplied by each other
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ii,ii->i")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (3,))])  # type: ignore

    def test_einsum_diag_dot_product_A2_B2(self) -> None:  # dot product of diagonals
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 3)), ("y", TensorProto.FLOAT, (3, 3))],
            [make_node("Einsum", ["x", "y"], ["z"], equation="ii,ii->")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NCdd(self) -> None:
        N, C = 3, 4
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C)), ("target", TensorProto.INT64, (N,))],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target"],
                    ["loss"],
                    reduction="none",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, (N,))])  # type: ignore

    def test_negative_log_likehood_shape_is_NC_with_weight(self) -> None:
        N, C = 3, 4
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C)),
                ("target", TensorProto.INT64, (N,)),
                ("weight", TensorProto.FLOAT, (C,)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target", "weight"],
                    ["loss"],
                    reduction="none",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, (N,))])  # type: ignore

    def test_negative_log_likehood_shape_is_NC_reduction_mean(self) -> None:
        N, C = 3, 4
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, (N, C)), ("target", TensorProto.INT64, (N,))],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target"],
                    ["loss"],
                    reduction="mean",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NC_with_weight_reduction_mean(self) -> None:
        N, C = 3, 4
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C)),
                ("target", TensorProto.INT64, (N,)),
                ("weight", TensorProto.FLOAT, (C,)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target", "weight"],
                    ["loss"],
                    reduction="mean",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2(self) -> None:
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C, d1, d2)),
                ("target", TensorProto.INT64, (N, d1, d2)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target"],
                    ["loss"],
                    reduction="none",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, (N, d1, d2))])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2_with_weight(self) -> None:
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C, d1, d2)),
                ("target", TensorProto.INT64, (N, d1, d2)),
                ("weight", TensorProto.FLOAT, (C,)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target", "weight"],
                    ["loss"],
                    reduction="none",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, (N, d1, d2))])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2_reduction_sum(self) -> None:
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C, d1, d2)),
                ("target", TensorProto.INT64, (N, d1, d2)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target"],
                    ["loss"],
                    reduction="sum",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_shape_is_NCd1d2_with_weight_reduction_mean(
        self,
    ) -> None:
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C, d1, d2)),
                ("target", TensorProto.INT64, (N, d1, d2)),
                ("weight", TensorProto.FLOAT, (C,)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target", "weight"],
                    ["loss"],
                    reduction="mean",
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("loss", TensorProto.FLOAT, ())])  # type: ignore

    def test_negative_log_likehood_input_target_shape_mismatch(self) -> None:
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, d1, d2)),
                ("target", TensorProto.INT64, (N, d1 + 1, d2)),
                ("weight", TensorProto.FLOAT, (C,)),
                ("loss", TensorProto.FLOAT, ()),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target", "weight"],
                    ["loss"],
                    reduction="mean",
                )
            ],
            [],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    def test_negative_log_likehood_input_weight_shape_mismatch(self) -> None:
        N, C, d1, d2 = 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("input", TensorProto.FLOAT, (N, C, d1, d2)),
                ("target", TensorProto.INT64, (N, d1, d2)),
                ("weight", TensorProto.FLOAT, (C + 1,)),
                ("loss", TensorProto.FLOAT, (N, d1, d2)),
            ],
            [
                make_node(
                    "NegativeLogLikelihoodLoss",
                    ["input", "target", "weight"],
                    ["loss"],
                    reduction="none",
                )
            ],
            [],
        )
        self.assertRaises(checker.ValidationError, self._inferred, graph)

    def test_softmax_cross_entropy_none(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)), ("y", TensorProto.FLOAT, (2,))],
            [make_node("SoftmaxCrossEntropyLoss", ["x", "y"], ["z"], reduction="none")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2,))])  # type: ignore

    def test_softmax_cross_entropy_mean(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (2, 3)), ("y", TensorProto.FLOAT, (2,))],
            [make_node("SoftmaxCrossEntropyLoss", ["x", "y"], ["z"], reduction="mean")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, ())])  # type: ignore

    def test_softmax_cross_entropy_none_NCD1D2(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (2, 3, 5, 8)),
                ("y", TensorProto.FLOAT, (2, 5, 8)),
            ],
            [make_node("SoftmaxCrossEntropyLoss", ["x", "y"], ["z"], reduction="none")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, (2, 5, 8))])  # type: ignore

    def test_softmax_cross_entropy_mean_NCD1D2(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (2, 3, 4, 5)),
                ("y", TensorProto.FLOAT, (2, 4, 5)),
            ],
            [make_node("SoftmaxCrossEntropyLoss", ["x", "y"], ["z"], reduction="mean")],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("z", TensorProto.FLOAT, ())])  # type: ignore

    def test_celu_function_output_shape(self) -> None:
        graph = self._make_graph(
            [("X", TensorProto.FLOAT, (25, 48, 16, 16))],
            [make_node("Celu", ["X"], ["Y"], alpha=2.0)],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("Y", TensorProto.FLOAT, (25, 48, 16, 16))]
        )

    def prepare_input_initializer_tensors(self, initializer_shape, input_shape):  # type: ignore
        nodes = [make_node("Add", ["x", "y"], "z")]
        if initializer_shape is None:
            initializer = []  # type: ignore
        else:
            size = 1
            for d in initializer_shape:
                size = size * d
            vals = [0.0 for i in range(size)]
            initializer = [
                make_tensor("x", TensorProto.FLOAT, initializer_shape, vals),  # type: ignore
                make_tensor("y", TensorProto.FLOAT, initializer_shape, vals),
            ]
        if input_shape is None:
            inputs = []  # type: ignore
        else:
            inputs = [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),  # type: ignore
                helper.make_tensor_value_info("y", TensorProto.FLOAT, input_shape),
            ]

        graph = helper.make_graph(
            nodes,
            "test",
            inputs=inputs,
            outputs=[],
            initializer=initializer,
            value_info=[],
        )
        return helper.make_model(graph)

    def test_infer_with_initializer_without_input_above_ir4(self) -> None:
        # This is for testing IR>=4: some tensors can only exist in initializer and not in input
        # So shape_inference should make use of initializer shapes
        initializer_shape = (8, 7)
        original_model = self.prepare_input_initializer_tensors(initializer_shape, None)
        inferred_model = onnx.shape_inference.infer_shapes(
            original_model, strict_mode=True
        )

        # If shape inference fails, it will throw IndexError
        z_tenor = inferred_model.graph.value_info.pop()
        z_shape = (
            z_tenor.type.tensor_type.shape.dim[0].dim_value,
            z_tenor.type.tensor_type.shape.dim[1].dim_value,
        )
        assert z_shape == initializer_shape

    def test_infer_with_initializer_without_input_below_ir4(self) -> None:
        # This is for testing IR<4: tensors must exist both in initializer and input
        # So shape_inference should not make use of initializer shapes
        # Use (None, None) as empty input
        initializer_shape = (8, 7)
        input_shape = (None, None)
        original_model = self.prepare_input_initializer_tensors(
            initializer_shape, input_shape
        )
        original_model.ir_version = 3  # test ir_version < 4

        inferred_model = onnx.shape_inference.infer_shapes(
            original_model, strict_mode=True
        )
        z_tenor = inferred_model.graph.value_info.pop()
        z_shape = (
            z_tenor.type.tensor_type.shape.dim[0].dim_value,
            z_tenor.type.tensor_type.shape.dim[1].dim_value,
        )
        # If the input is not updated by the initializer, the output shape will keep empty (0, 0)
        assert z_shape == (0, 0)

    def test_infer_initializer_input_mismatch(self) -> None:
        # Catch error if initializer and input mismatch
        initializer_shape = (8, 7)
        input_shape = (4, 3)
        original_model = self.prepare_input_initializer_tensors(
            initializer_shape, input_shape
        )
        # Inferred shape and existing shape differ in dimension 0
        self.assertRaises(
            onnx.shape_inference.InferenceError,
            onnx.shape_inference.infer_shapes,
            original_model,
            strict_mode=True,
        )

    def test_infer_initializer_input_consistency_all_none(self) -> None:
        initializer_shape = (8, 7)
        input_shape = (None, None)  # accepatble
        original_model = self.prepare_input_initializer_tensors(
            initializer_shape, input_shape
        )

        onnx.shape_inference.infer_shapes(original_model, strict_mode=True)

    def test_infer_initializer_input_consistency_single_none(self) -> None:
        initializer_shape = (8, 7)
        input_shape = (None, 7)  # accepatble
        original_model = self.prepare_input_initializer_tensors(
            initializer_shape, input_shape
        )

        onnx.shape_inference.infer_shapes(original_model, strict_mode=True)

    def test_infer_initializer_input_consistency_differnt_rank(self) -> None:
        initializer_shape = (8, 7, 9)
        input_shape = (None, 7)  # accepatble
        original_model = self.prepare_input_initializer_tensors(
            initializer_shape, input_shape
        )
        # Inferred shape and existing shape differ in rank: (3) vs (2)
        self.assertRaises(
            onnx.shape_inference.InferenceError,
            onnx.shape_inference.infer_shapes,
            original_model,
            strict_mode=True,
        )

    def test_infer_initializer_input_consistency_all_none_serialized(self) -> None:
        # Reuse test_infer_initializer_input_consistency_all_none test case and check with
        # Serialized model
        initializer_shape = (8, 7)
        input_shape = (None, None)  # accepatble
        original_model = self.prepare_input_initializer_tensors(
            initializer_shape, input_shape
        )

        onnx.shape_inference.infer_shapes(
            original_model.SerializeToString(), strict_mode=True
        )

    def test_trilu_upper(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("k", TensorProto.INT64, ())],
            [make_node("Trilu", ["x", "k"], ["y"])],
            [],
            initializer=[make_tensor("k", TensorProto.INT64, (), (2,))],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 4, 5))])  # type: ignore

    def test_trilu_lower(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3, 4, 5)), ("k", TensorProto.INT64, ())],
            [make_node("Trilu", ["x", "k"], ["y"], upper=0)],
            [],
            initializer=[make_tensor("k", TensorProto.INT64, (), (10,))],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.FLOAT, (3, 4, 5))])  # type: ignore

    def test_trilu_upper_zero(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT64, (0, 5)), ("k", TensorProto.INT64, ())],
            [make_node("Trilu", ["x", "k"], ["y"], upper=1)],
            [],
            initializer=[make_tensor("k", TensorProto.INT64, (), (5,))],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.INT64, (0, 5))])  # type: ignore

    def test_trilu_lower_one(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.INT32, (3, 1, 5))],
            [make_node("Trilu", ["x"], ["y"], upper=0)],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.INT32, (3, 1, 5))])  # type: ignore

    def test_batch_norm_train(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5, 6, 7)),
                ("scale", TensorProto.FLOAT, (4,)),
                ("b", TensorProto.FLOAT, (4,)),
                ("input_mean", TensorProto.FLOAT, (4,)),
                ("input_var", TensorProto.FLOAT, (4,)),
            ],
            [
                make_node(
                    "BatchNormalization",
                    ["x", "scale", "b", "input_mean", "input_var"],
                    ["out", "output_mean", "output_var"],
                    training_mode=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("out", TensorProto.FLOAT, (3, 4, 5, 6, 7)),  # type: ignore
                make_tensor_value_info("output_mean", TensorProto.FLOAT, (4,)),  # type: ignore
                make_tensor_value_info("output_var", TensorProto.FLOAT, (4,)),  # type: ignore
            ],
        )

    def test_batch_norm_train_dim_param(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, "C", 5, 6, 7)),
                ("scale", TensorProto.FLOAT, ("C",)),
                ("b", TensorProto.FLOAT, ("C",)),
                ("input_mean", TensorProto.FLOAT, ("C",)),
                ("input_var", TensorProto.FLOAT, ("C",)),
            ],
            [
                make_node(
                    "BatchNormalization",
                    ["x", "scale", "b", "input_mean", "input_var"],
                    ["out", "output_mean", "output_var"],
                    training_mode=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("out", TensorProto.FLOAT, (3, "C", 5, 6, 7)),  # type: ignore
                make_tensor_value_info("output_mean", TensorProto.FLOAT, ("C",)),  # type: ignore
                make_tensor_value_info("output_var", TensorProto.FLOAT, ("C",)),  # type: ignore
            ],
        )

    def test_batch_norm_train_with_diff_type(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT16, (3, 4, 5, 6, 7)),
                ("scale", TensorProto.FLOAT16, (4,)),
                ("b", TensorProto.FLOAT16, (4,)),
                ("input_mean", TensorProto.FLOAT, (4,)),
                ("input_var", TensorProto.FLOAT, (4,)),
            ],
            [
                make_node(
                    "BatchNormalization",
                    ["x", "scale", "b", "input_mean", "input_var"],
                    ["out", "output_mean", "output_var"],
                    training_mode=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("out", TensorProto.FLOAT16, (3, 4, 5, 6, 7)),  # type: ignore
                make_tensor_value_info("output_mean", TensorProto.FLOAT, (4,)),  # type: ignore
                make_tensor_value_info("output_var", TensorProto.FLOAT, (4,)),  # type: ignore
            ],
        )

    def test_batch_norm_test(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, 5, 6, 7)),
                ("scale", TensorProto.FLOAT, (4,)),
                ("b", TensorProto.FLOAT, (4,)),
                ("input_mean", TensorProto.FLOAT, (4,)),
                ("input_var", TensorProto.FLOAT, (4,)),
            ],
            [
                make_node(
                    "BatchNormalization",
                    ["x", "scale", "b", "input_mean", "input_var"],
                    ["out"],
                    training_mode=0,
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.FLOAT, (3, 4, 5, 6, 7))])  # type: ignore

    def test_batch_norm_test_no_dim(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (3, 4, None, None, None)),
                ("scale", TensorProto.FLOAT, (4,)),
                ("b", TensorProto.FLOAT, (4,)),
                ("input_mean", TensorProto.FLOAT, (None,)),
                ("input_var", TensorProto.FLOAT, (4,)),
            ],
            [
                make_node(
                    "BatchNormalization",
                    ["x", "scale", "b", "input_mean", "input_var"],
                    ["out"],
                    training_mode=0,
                )
            ],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.FLOAT, (3, 4, None, None, None))])  # type: ignore

    def test_batch_norm_train_no_shape(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, None),
                ("scale", TensorProto.FLOAT, None),
                ("b", TensorProto.FLOAT, None),
                ("input_mean", TensorProto.FLOAT, ("C",)),
                ("input_var", TensorProto.FLOAT, ("C",)),
            ],
            [
                make_node(
                    "BatchNormalization",
                    ["x", "scale", "b", "input_mean", "input_var"],
                    ["out", "running_mean", "running_var"],
                    training_mode=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("out", TensorProto.FLOAT, None),  # type: ignore
                make_tensor_value_info("running_mean", TensorProto.FLOAT, ("C",)),  # type: ignore
                make_tensor_value_info("running_var", TensorProto.FLOAT, ("C",)),  # type: ignore
            ],
        )

    def test_nonzero(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (None,))],
            [make_node("NonZero", ["x"], ["out"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.INT64, (1, None))])  # type: ignore

    def test_nonzero_no_shape(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, None)], [make_node("NonZero", ["x"], ["out"])], []
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.INT64, (None, None))])  # type: ignore

    def test_nonzero_existing_dim_param(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, (3,))],
            [make_node("NonZero", ["x"], ["y"])],
            [make_tensor_value_info("y", TensorProto.INT64, (None, "NZ"))],
        )
        self._assert_inferred(graph, [make_tensor_value_info("y", TensorProto.INT64, (1, "NZ"))])  # type: ignore

    def test_nonzero_scalar(self) -> None:
        graph = self._make_graph(
            [("x", TensorProto.FLOAT, ())], [make_node("NonZero", ["x"], ["out"])], []
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.INT64, (0, None))])  # type: ignore

    def test_optional_construct_empty_tensor(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=[1, 2, 3]
        )
        optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
        optional_val_info = helper.make_value_info(
            name="output", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [], [make_node("Optional", [], ["output"], type=tensor_type_proto)], []
        )
        self._assert_inferred(graph, [optional_val_info])  # type: ignore

    def test_optional_construct_empty_sequence(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.INT32, shape=[1, 2, 3]
        )
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
        optional_val_info = helper.make_value_info(
            name="output_sequence", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [],
            [make_node("Optional", [], ["output_sequence"], type=sequence_type_proto)],
            [],
        )
        self._assert_inferred(graph, [optional_val_info])  # type: ignore

    def test_optional_construct_tensor(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=[2, 3, 4]
        )
        optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
        optional_val_info = helper.make_value_info(
            name="output", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [("input1", TensorProto.FLOAT, (2, 3, 4))],
            [make_node("Optional", ["input1"], ["output"])],
            [],
        )
        self._assert_inferred(graph, [optional_val_info])  # type: ignore

    def test_optional_construct_sequence(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.INT64, shape=[2, 3, 0]
        )
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        sequence_val_info = helper.make_value_info(
            name="input_sequence", type_proto=sequence_type_proto
        )
        optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
        optional_val_info = helper.make_value_info(
            name="output_sequence", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [("input1", TensorProto.INT64, (2, 3, 0))],
            [
                make_node("SequenceConstruct", ["input1"], ["input_sequence"]),
                make_node("Optional", ["input_sequence"], ["output_sequence"]),
            ],
            [],
        )
        self._assert_inferred(graph, [sequence_val_info, optional_val_info])  # type: ignore

    def test_optional_tensor_has_element(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=[2, 3, 4]
        )
        optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
        optional_val_info = helper.make_value_info(
            name="sequence", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [("input1", TensorProto.FLOAT, (2, 3, 4))],
            [
                make_node("Optional", ["input1"], ["sequence"]),
                make_node("OptionalHasElement", ["sequence"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [optional_val_info, make_tensor_value_info("output", TensorProto.BOOL, ())],
        )  # type: ignore

    def test_optional_sequence_has_element(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.FLOAT, shape=[0, 3, 4]
        )
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        sequence_val_info = helper.make_value_info(
            name="sequence", type_proto=sequence_type_proto
        )
        optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
        optional_val_info = helper.make_value_info(
            name="optional", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [("input1", TensorProto.FLOAT, (0, 3, 4))],
            [
                make_node("SequenceConstruct", ["input1"], ["sequence"]),
                make_node("Optional", ["sequence"], ["optional"]),
                make_node("OptionalHasElement", ["optional"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                sequence_val_info,
                optional_val_info,
                make_tensor_value_info("output", TensorProto.BOOL, ()),
            ],
        )  # type: ignore

    def test_tensor_get_element(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.DOUBLE, shape=[2, 1, 4]
        )
        output_tensor_val_info = helper.make_value_info(
            name="output", type_proto=tensor_type_proto
        )
        graph = self._make_graph(
            [("input", TensorProto.DOUBLE, (2, 1, 4))],
            [
                make_node("OptionalGetElement", ["input"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(graph, [output_tensor_val_info])  # type: ignore

    @parameterized.expand(all_versions_for("StringSplit"))
    def test_string_split_basic(self, _, version) -> None:
        substrings = make_tensor_value_info(
            "substrings",
            TensorProto.STRING,
            (2, None),
        )
        length = make_tensor_value_info("length", TensorProto.INT64, (2,))
        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, (2,)),
            ],
            [make_node("StringSplit", ["x"], ["substrings", "length"])],
            [substrings, length],
        )
        self._assert_inferred(
            graph,
            [substrings, length],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("StringSplit"))
    def test_string_split_symbolic(self, _, version) -> None:
        substrings = make_tensor_value_info(
            "substrings",
            TensorProto.STRING,
            ("A", None),
        )
        length = make_tensor_value_info("length", TensorProto.INT64, ("A",))
        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, ("A",)),
            ],
            [make_node("StringSplit", ["x"], ["substrings", "length"])],
            [substrings, length],
        )
        self._assert_inferred(
            graph,
            [substrings, length],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("StringSplit"))
    def test_string_split_nested(self, _, version) -> None:
        substrings = make_tensor_value_info(
            "substrings", TensorProto.STRING, (2, 4, 3, None)
        )
        length = make_tensor_value_info("length", TensorProto.INT64, (2, 4, 3))
        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, (2, 4, 3)),
            ],
            [make_node("StringSplit", ["x"], ["substrings", "length"], maxsplit=2)],
            [substrings, length],
        )
        self._assert_inferred(
            graph,
            [substrings, length],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("StringSplit"))
    def test_string_split_zero_dimensional_input(self, _, version) -> None:
        substrings = make_tensor_value_info("substrings", TensorProto.STRING, (None,))
        length = make_tensor_value_info("length", TensorProto.INT64, ())

        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, ()),
            ],
            [make_node("StringSplit", ["x"], ["substrings", "length"], maxsplit=2)],
            [substrings, length],
        )
        self._assert_inferred(
            graph,
            [substrings, length],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(all_versions_for("StringSplit"))
    def test_string_split_empty_input(self, _, version) -> None:
        substrings = make_tensor_value_info(
            "substrings", TensorProto.STRING, ("M", 3, 0, None)
        )
        length = make_tensor_value_info("length", TensorProto.INT64, ("M", 3, 0))

        graph = self._make_graph(
            [
                ("x", TensorProto.STRING, ("M", 3, 0)),
            ],
            [make_node("StringSplit", ["x"], ["substrings", "length"], maxsplit=2)],
            [substrings, length],
        )
        self._assert_inferred(
            graph,
            [substrings, length],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    def test_optional_tensor_get_element(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.DOUBLE, shape=[2, 1, 4]
        )
        tensor_val_into = helper.make_value_info(
            name="output", type_proto=tensor_type_proto
        )
        optional_type_proto = helper.make_optional_type_proto(tensor_type_proto)
        optional_val_info = helper.make_value_info(
            name="optional", type_proto=optional_type_proto
        )
        graph = self._make_graph(
            [("input1", TensorProto.DOUBLE, (2, 1, 4))],
            [
                make_node("Optional", ["input1"], ["optional"]),
                make_node("OptionalGetElement", ["optional"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(graph, [optional_val_info, tensor_val_into])  # type: ignore

    def test_optional_sequence_get_element(self) -> None:
        tensor_type_proto = helper.make_tensor_type_proto(
            elem_type=TensorProto.INT32, shape=[2, 0, 4]
        )
        sequence_type_proto = helper.make_sequence_type_proto(tensor_type_proto)
        sequence_val_into = helper.make_value_info(
            name="sequence", type_proto=sequence_type_proto
        )
        optional_type_proto = helper.make_optional_type_proto(sequence_type_proto)
        optional_val_info = helper.make_value_info(
            name="optional", type_proto=optional_type_proto
        )
        output_val_into = helper.make_value_info(
            name="output", type_proto=sequence_type_proto
        )
        graph = self._make_graph(
            [("input1", TensorProto.INT32, (2, 0, 4))],
            [
                make_node("SequenceConstruct", ["input1"], ["sequence"]),
                make_node("Optional", ["sequence"], ["optional"]),
                make_node("OptionalGetElement", ["optional"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(graph, [optional_val_info, sequence_val_into, output_val_into])  # type: ignore

    def test_where_bfloat(self) -> None:
        graph = self._make_graph(
            [
                ("cond", TensorProto.BOOL, (10,)),
                ("x", TensorProto.BFLOAT16, (10,)),
                ("y", TensorProto.BFLOAT16, (10,)),
            ],
            [make_node("Where", ["cond", "x", "y"], ["out"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("out", TensorProto.BFLOAT16, (10,))])  # type: ignore

    def test_parse_data_with_unsupported_tensor_type(self) -> None:
        model = helper.make_model(
            graph=helper.make_graph(
                name="graph_with_unsupported_type",
                inputs=[],
                outputs=[
                    helper.make_tensor_value_info("y", TensorProto.FLOAT, shape=None)
                ],
                nodes=[make_node("ConstantOfShape", ["x"], ["y"])],
                # ConstantOfShape only accepts np.int64 instead of np.int32
                initializer=[
                    numpy_helper.from_array(np.array([4, 3], dtype=np.int32), name="x")
                ],
            )
        )
        # Strict shape inference should catch this invalid type error (int32 is not supported)
        self.assertRaises(
            onnx.shape_inference.InferenceError,
            onnx.shape_inference.infer_shapes,
            model,
            strict_mode=True,
        )
        # Even nornmal shape inference should not produce any invalid shape due to wrong type for ParseData
        inferred_model = onnx.shape_inference.infer_shapes(model)
        self.assertFalse(
            inferred_model.graph.output[0].type.tensor_type.HasField("shape")
        )

    def test_parse_data_with_undefined_tensor_type(self) -> None:
        model = helper.make_model(
            graph=helper.make_graph(
                name="graph_with_undefined_type",
                inputs=[],
                outputs=[
                    helper.make_tensor_value_info("y", TensorProto.FLOAT, shape=None)
                ],
                nodes=[make_node("ConstantOfShape", ["x"], ["y"])],
                initializer=[
                    numpy_helper.from_array(np.array([4, 3], dtype=np.int64), name="x")
                ],
            )
        )
        # Hardcode the tensor type as UNDEFINED to test catching undefined type error
        model.graph.initializer[0].data_type = TensorProto.UNDEFINED
        # Strict shape inference should catch this undefined type error
        self.assertRaises(
            onnx.shape_inference.InferenceError,
            onnx.shape_inference.infer_shapes,
            model,
            strict_mode=True,
        )
        # Even nornmal shape inference should not produce any invalid shape due to undefined type for ParseData
        inferred_model = onnx.shape_inference.infer_shapes(model)
        self.assertFalse(
            inferred_model.graph.output[0].type.tensor_type.HasField("shape")
        )

        graph = self._make_graph(
            [("x", TensorProto.UINT8, (1, 0, 0)), ("shape", TensorProto.INT64, (3,))],
            [make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (3,), (0, 1, 1))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.UINT8, (0, 1, 1))]
        )

    def test_affinegrid_2d(self) -> None:
        N, C, H, W = 2, 3, 4, 5
        graph = self._make_graph(
            [
                ("theta", TensorProto.FLOAT, (N, 2, 3)),
                ("size", TensorProto.INT64, (4,)),
            ],
            [
                make_node(
                    "AffineGrid",
                    ["theta", "size"],
                    ["grid"],
                    align_corners=1,
                )
            ],
            [],
            initializer=[make_tensor("size", TensorProto.INT64, (4,), (N, C, H, W))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("grid", TensorProto.FLOAT, (N, H, W, 2))]
        )  # type: ignore

    def test_affinegrid_3d(self) -> None:
        N, C, D, H, W = 2, 3, 4, 5, 6
        graph = self._make_graph(
            [
                ("theta", TensorProto.FLOAT, (N, 3, 4)),
                ("size", TensorProto.INT64, (5,)),
            ],
            [
                make_node(
                    "AffineGrid",
                    ["theta", "size"],
                    ["grid"],
                )
            ],
            [],
            initializer=[make_tensor("size", TensorProto.INT64, (5,), (N, C, D, H, W))],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("grid", TensorProto.FLOAT, (N, D, H, W, 3))]
        )  # type: ignore

    def test_gridsample_2d(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (1, 1, 3, 3)),
                ("grid", TensorProto.INT64, (1, 3, 3, 2)),
            ],
            [
                make_node(
                    "GridSample",
                    ["x", "grid"],
                    ["y"],
                    mode="nearest",
                    padding_mode="border",
                    align_corners=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 1, 3, 3))]
        )  # type: ignore

    def test_gridsample_3d(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, (1, 1, 3, 3, 3)),
                ("grid", TensorProto.INT64, (1, 3, 2, 3, 3)),
            ],
            [
                make_node(
                    "GridSample",
                    ["x", "grid"],
                    ["y"],
                    mode="nearest",
                    padding_mode="border",
                    align_corners=1,
                )
            ],
            [],
        )
        self._assert_inferred(
            graph, [make_tensor_value_info("y", TensorProto.FLOAT, (1, 1, 3, 2, 3))]
        )  # type: ignore

    def test_gridsample_2d_defaults(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", "H", "W")),
                ("grid", TensorProto.FLOAT, ("N", "H_out", "W_out", 2)),
            ],
            [make_node("GridSample", ["x", "grid"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "y", TensorProto.FLOAT, ("N", "C", "H_out", "W_out")
                )
            ],
        )  # type: ignore

    def test_gridsample_3d_defaults(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", "D", "H", "W")),
                ("grid", TensorProto.FLOAT, ("N", "D_out", "H_out", "W_out", 3)),
            ],
            [make_node("GridSample", ["x", "grid"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "y", TensorProto.FLOAT, ("N", "C", "D_out", "H_out", "W_out")
                )
            ],
        )  # type: ignore

    def test_gridsample_2d_no_dim(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", None, None)),
                ("grid", TensorProto.FLOAT, ("N", None, None, 2)),
            ],
            [
                make_node(
                    "GridSample",
                    ["x", "grid"],
                    ["y"],
                    mode="linear",
                    padding_mode="border",
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, ("N", "C", None, None))],
        )  # type: ignore

    def test_gridsample_3d_no_dim(self) -> None:
        graph = self._make_graph(
            [
                ("x", TensorProto.FLOAT, ("N", "C", None, None, None)),
                ("grid", TensorProto.FLOAT, ("N", None, None, None, 3)),
            ],
            [
                make_node(
                    "GridSample",
                    ["x", "grid"],
                    ["y"],
                    mode="linear",
                    padding_mode="border",
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info(
                    "y", TensorProto.FLOAT, ("N", "C", None, None, None)
                )
            ],
        )  # type: ignore

    def test_sequence_map_identity_known_dims(self):
        input_value_infos = [
            make_tensor_value_info("input", TensorProto.FLOAT, (220, 220, 3))
        ]
        output_value_infos = [
            make_tensor_value_info("output", TensorProto.FLOAT, (220, 220, 3))
        ]
        body_graph = helper.make_graph(
            [make_node("Identity", ["input"], ["output"])],
            "body_graph",
            input_value_infos,
            output_value_infos,
        )
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (220, 220, 3)),
                ("input2", TensorProto.FLOAT, (220, 220, 3)),
                ("input3", TensorProto.FLOAT, (220, 220, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceMap", ["in_sequence"], ["out_sequence"], body=body_graph
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (220, 220, 3)
                ),
                make_tensor_sequence_value_info(
                    "out_sequence", TensorProto.FLOAT, (220, 220, 3)
                ),
            ],
        )  # type: ignore

    def test_sequence_map_identity_unknown_dims(self):
        input_value_infos = [
            make_tensor_value_info("input", TensorProto.FLOAT, ("H", "W", 3))
        ]
        output_value_infos = [
            make_tensor_value_info("output", TensorProto.FLOAT, ("H", "W", 3))
        ]
        body_graph = helper.make_graph(
            [make_node("Identity", ["input"], ["output"])],
            "body_graph",
            input_value_infos,
            output_value_infos,
        )
        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (200, 300, 3)),
                ("input2", TensorProto.FLOAT, (100, 200, 3)),
                ("input3", TensorProto.FLOAT, (5, 1, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceMap", ["in_sequence"], ["out_sequence"], body=body_graph
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (None, None, 3)
                ),
                make_tensor_sequence_value_info(
                    "out_sequence", TensorProto.FLOAT, (None, None, 3)
                ),
            ],
        )  # type: ignore

    def test_sequence_map_slice_outs_known_dims(self):
        body_graph = helper.make_graph(
            nodes=[
                make_node("Slice", ["x", "starts1", "ends1", "axes", ""], ["y1"]),
                make_node("Slice", ["x", "starts2", "ends2", "axes", ""], ["y2"]),
            ],
            name="body_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, ("H", "W", 3)
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "y1", onnx.TensorProto.FLOAT, (10, 20, 3)
                ),
                onnx.helper.make_tensor_value_info(
                    "y2", onnx.TensorProto.FLOAT, (30, 40, 3)
                ),
            ],
            initializer=[
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
                make_tensor("starts1", TensorProto.INT64, (2,), (0, 0)),
                make_tensor("ends1", TensorProto.INT64, (2,), (10, 20)),
                make_tensor("starts2", TensorProto.INT64, (2,), (0, 0)),
                make_tensor("ends2", TensorProto.INT64, (2,), (30, 40)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (220, 310, 3)),
                ("input2", TensorProto.FLOAT, (110, 210, 3)),
                ("input3", TensorProto.FLOAT, (90, 110, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceMap",
                    ["in_sequence"],
                    ["out_sequence1", "out_sequence2"],
                    body=body_graph,
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (None, None, 3)
                ),
                make_tensor_sequence_value_info(
                    "out_sequence1", TensorProto.FLOAT, (10, 20, 3)
                ),
                make_tensor_sequence_value_info(
                    "out_sequence2", TensorProto.FLOAT, (30, 40, 3)
                ),
            ],
        )  # type: ignore

    def test_sequence_map_slice_outs_unknown_dims(self):
        body_graph = helper.make_graph(
            nodes=[
                make_node("Slice", ["x", "starts1", "ends1", "axes", ""], ["y1"]),
                make_node("Slice", ["x", "starts2", "ends2", "axes", ""], ["y2"]),
            ],
            name="body_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, ("H", "W", 3)
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "y1", onnx.TensorProto.FLOAT, ("H1", "W1", 3)
                ),
                onnx.helper.make_tensor_value_info(
                    "y2", onnx.TensorProto.FLOAT, ("H2", "W2", 3)
                ),
            ],
            initializer=[
                make_tensor("axes", TensorProto.INT64, (2,), (0, 1)),
                make_tensor("starts1", TensorProto.INT64, (2,), (0, 0)),
                make_tensor("ends1", TensorProto.INT64, (2,), (10, 20)),
                make_tensor("starts2", TensorProto.INT64, (2,), (0, 0)),
                make_tensor("ends2", TensorProto.INT64, (2,), (30, 40)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (220, 310, 3)),
                ("input2", TensorProto.FLOAT, (110, 210, 3)),
                ("input3", TensorProto.FLOAT, (90, 110, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node(
                    "SequenceMap",
                    ["in_sequence"],
                    ["out_sequence1", "out_sequence2"],
                    body=body_graph,
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (None, None, 3)
                ),
                make_tensor_sequence_value_info(
                    "out_sequence1", TensorProto.FLOAT, (None, None, 3)
                ),
                make_tensor_sequence_value_info(
                    "out_sequence2", TensorProto.FLOAT, (None, None, 3)
                ),
            ],
        )  # type: ignore

    def test_sequence_map_different_tensor_type(self):
        body_graph = helper.make_graph(
            nodes=[make_node("Shape", ["x"], ["shape"])],
            name="body_graph",
            inputs=[
                onnx.helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, ("H", "W", "C")
                )
            ],
            outputs=[
                onnx.helper.make_tensor_value_info(
                    "shape", onnx.TensorProto.INT64, (3,)
                )
            ],
        )  # type: ignore

        graph = self._make_graph(
            [
                ("input1", TensorProto.FLOAT, (220, 310, 3)),
                ("input2", TensorProto.FLOAT, (110, 210, 3)),
                ("input3", TensorProto.FLOAT, (90, 110, 3)),
            ],
            [
                make_node(
                    "SequenceConstruct", ["input1", "input2", "input3"], ["in_sequence"]
                ),
                make_node("SequenceMap", ["in_sequence"], ["shapes"], body=body_graph),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_sequence_value_info(
                    "in_sequence", TensorProto.FLOAT, (None, None, 3)
                ),
                make_tensor_sequence_value_info("shapes", TensorProto.INT64, (3,)),
            ],
        )  # type: ignore

    def test_hammingwindow(self):
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (), (10,)),
                ),
                make_node("HammingWindow", ["shape"], ["y"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, ()),
                make_tensor_value_info("y", TensorProto.FLOAT, (10,)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (), (10,)),
                ),
                make_node("HammingWindow", ["shape"], ["y"], periodic=0),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, ()),
                make_tensor_value_info("y", TensorProto.FLOAT, (10,)),
            ],
        )  # type: ignore

    def test_hannwindow(self):
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (), (10,)),
                ),
                make_node("HannWindow", ["shape"], ["y"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, ()),
                make_tensor_value_info("y", TensorProto.FLOAT, (10,)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (), (10,)),
                ),
                make_node("HannWindow", ["shape"], ["y"], periodic=0),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, ()),
                make_tensor_value_info("y", TensorProto.FLOAT, (10,)),
            ],
        )  # type: ignore

    def test_blackmanwindow(self):
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (), (10,)),
                ),
                make_node("BlackmanWindow", ["shape"], ["y"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, ()),
                make_tensor_value_info("y", TensorProto.FLOAT, (10,)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["shape"],
                    value=make_tensor("shape", TensorProto.INT64, (), (10,)),
                ),
                make_node("BlackmanWindow", ["shape"], ["y"], periodic=0),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("shape", TensorProto.INT64, ()),
                make_tensor_value_info("y", TensorProto.FLOAT, (10,)),
            ],
        )  # type: ignore

    @parameterized.expand(
        [
            (
                name,
                version,
                test_aspect,
                input_shape,
                axis,
                onesided,
                inverse,
                expected_shape,
            )
            for (name, version), (
                test_aspect,
                input_shape,
                axis,
                onesided,
                inverse,
                expected_shape,
            ) in itertools.product(
                all_versions_for("DFT"),
                (
                    ("reals_default_axis", (2, 5, 1), None, None, None, (2, 5, 2)),
                    ("reals_axis_0", (3, 5, 10, 1), 0, 0, 0, (3, 5, 10, 2)),
                    ("reals_axis_1", (3, 5, 10, 1), 1, 0, 0, (3, 5, 10, 2)),
                    ("reals_axis_2", (3, 5, 10, 1), 2, 0, 0, (3, 5, 10, 2)),
                    ("reals_axis_neg", (3, 5, 10, 1), -2, 0, 0, (3, 5, 10, 2)),
                    ("reals_axis_0_onesided", (3, 5, 10, 1), 0, 1, 0, (2, 5, 10, 2)),
                    ("reals_axis_1_onesided", (3, 5, 10, 1), 1, 1, 0, (3, 3, 10, 2)),
                    ("reals_axis_2_onesided", (3, 5, 10, 1), 2, 1, 0, (3, 5, 6, 2)),
                    ("reals_axis_neg_onesided", (3, 5, 10, 1), -2, 1, 0, (3, 5, 6, 2)),
                    ("complex_default_axis", (2, 5, 2), None, None, None, (2, 5, 2)),
                    ("complex_onesided", (2, 5, 2), 1, 1, None, (2, 3, 2)),
                    ("real_inverse", (2, 5, 1), 1, None, 1, (2, 5, 2)),
                    ("complex_inverse", (2, 5, 2), 1, None, 1, (2, 5, 2)),
                ),
            )
        ]
    )
    def test_dft(
        self,
        _: str,
        version: int,
        _test_aspect: str,
        input_shape: tuple[int],
        axis: int | None,
        onesided: int | None,
        inverse: int | None,
        expected_shape: tuple[int],
    ) -> None:
        # Build the attributes for different opset versions
        attributes = {}
        if onesided is not None:
            attributes["onesided"] = onesided
        if inverse is not None:
            attributes["inverse"] = inverse

        if version < 20:
            if axis is not None:
                attributes["axis"] = axis
            nodes = [make_node("DFT", ["input", ""], ["output"], **attributes)]  # type: ignore[arg-type]
            value_infos = []
        else:
            assert version >= 20
            if axis is not None:
                nodes = [
                    make_node(
                        "Constant",
                        [],
                        ["axis"],
                        value=make_tensor("axis", TensorProto.INT64, (), (axis,)),
                    ),
                    make_node("DFT", ["input", "", "axis"], ["output"], **attributes),  # type: ignore[arg-type]
                ]
                value_infos = [make_tensor_value_info("axis", TensorProto.INT64, ())]
            else:
                nodes = [
                    make_node("DFT", ["input", "", ""], ["output"], **attributes),  # type: ignore[arg-type]
                ]
                value_infos = []

        # Construct the graph
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        input_shape,
                        np.ones(input_shape, dtype=np.float32).flatten(),
                    ),
                ),
                *nodes,
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                *value_infos,
                make_tensor_value_info("output", TensorProto.FLOAT, expected_shape),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(
        [
            (
                name,
                version,
                test_aspect,
                input_shape,
                axis,
                onesided,
                inverse,
                expected_shape,
            )
            for (name, version), (
                test_aspect,
                input_shape,
                axis,
                onesided,
                inverse,
                expected_shape,
            ) in itertools.product(
                all_versions_for("DFT"),
                (
                    ("reals_default_axis", (2, 5, 1), None, None, None, (2, 42, 2)),
                    ("reals_axis_0", (3, 5, 10, 1), 0, 0, 0, (42, 5, 10, 2)),
                    ("reals_axis_1", (3, 5, 10, 1), 1, 0, 0, (3, 42, 10, 2)),
                    ("reals_axis_2", (3, 5, 10, 1), 2, 0, 0, (3, 5, 42, 2)),
                    ("reals_axis_neg", (3, 5, 10, 1), -2, 0, 0, (3, 5, 42, 2)),
                    ("reals_axis_0_onesided", (3, 5, 10, 1), 0, 1, 0, (22, 5, 10, 2)),
                    ("reals_axis_1_onesided", (3, 5, 10, 1), 1, 1, 0, (3, 22, 10, 2)),
                    ("reals_axis_2_onesided", (3, 5, 10, 1), 2, 1, 0, (3, 5, 22, 2)),
                    ("reals_axis_neg_onesided", (3, 5, 10, 1), -2, 1, 0, (3, 5, 22, 2)),
                    ("complex_default_axis", (2, 5, 2), None, None, None, (2, 42, 2)),
                    ("complex_onesided", (2, 5, 2), 1, 1, None, (2, 22, 2)),
                    ("real_inverse", (2, 5, 1), 1, None, 1, (2, 42, 2)),
                    ("complex_inverse", (2, 5, 2), 1, None, 1, (2, 42, 2)),
                ),
            )
        ]
    )
    def test_dft_dft_length(
        self,
        _: str,
        version: int,
        _test_aspect: str,
        input_shape: tuple[int],
        axis: int | None,
        onesided: int | None,
        inverse: int | None,
        expected_shape: tuple[int],
    ) -> None:
        # Build the attributes for different opset versions
        attributes = {}
        if onesided is not None:
            attributes["onesided"] = onesided
        if inverse is not None:
            attributes["inverse"] = inverse

        dft_length = 42

        if version < 20:
            if axis is not None:
                attributes["axis"] = axis
            nodes = [
                make_node(
                    "Constant",
                    [],
                    ["dft_length"],
                    value=make_tensor(
                        "dft_length", TensorProto.INT64, (), (dft_length,)
                    ),
                ),
                make_node("DFT", ["input", "dft_length"], ["output"], **attributes),  # type: ignore[arg-type]
            ]
            value_infos = [make_tensor_value_info("dft_length", TensorProto.INT64, ())]
        else:
            assert version >= 20
            if axis is not None:
                nodes = [
                    make_node(
                        "Constant",
                        [],
                        ["axis"],
                        value=make_tensor("axis", TensorProto.INT64, (), (axis,)),
                    ),
                    make_node(
                        "Constant",
                        [],
                        ["dft_length"],
                        value=make_tensor(
                            "dft_length", TensorProto.INT64, (), (dft_length,)
                        ),
                    ),
                    make_node("DFT", ["input", "dft_length", "axis"], ["output"], **attributes),  # type: ignore[arg-type]
                ]
                value_infos = [
                    make_tensor_value_info("dft_length", TensorProto.INT64, ()),
                    make_tensor_value_info("axis", TensorProto.INT64, ()),
                ]
            else:
                nodes = [
                    make_node(
                        "Constant",
                        [],
                        ["dft_length"],
                        value=make_tensor(
                            "dft_length", TensorProto.INT64, (), (dft_length,)
                        ),
                    ),
                    make_node("DFT", ["input", "dft_length", ""], ["output"], **attributes),  # type: ignore[arg-type]
                ]
                value_infos = [
                    make_tensor_value_info("dft_length", TensorProto.INT64, ())
                ]

        # Construct the graph
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        input_shape,
                        np.ones(input_shape, dtype=np.float32).flatten(),
                    ),
                ),
                *nodes,
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                *value_infos,
                make_tensor_value_info("output", TensorProto.FLOAT, expected_shape),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, version)],
        )

    @parameterized.expand(
        [
            ("last", 3),
            ("last_negative", -1),
            ("out_of_range", 4),
            ("out_of_range_negative", -5),
        ]
    )
    def test_dft_invalid_axis_opset17(self, _: str, axis: int) -> None:
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        (2, 5, 5, 2),
                        np.ones((2, 5, 5, 2), dtype=np.float32).flatten(),
                    ),
                ),
                make_node("DFT", ["input", ""], ["output"], onesided=1, axis=axis),
            ],
            [],
        )
        with self.assertRaises(onnx.shape_inference.InferenceError):
            self._assert_inferred(
                graph,
                [
                    make_tensor_value_info("input", TensorProto.FLOAT, (2, 5, 5, 2)),
                    make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 5, 2)),
                ],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 17)],
            )

    @parameterized.expand(
        [
            ("last", 3),
            ("last_negative", -1),
            ("out_of_range", 4),
            ("out_of_range_negative", -5),
        ]
    )
    def test_dft_invalid_axis_opset20(self, _: str, axis: int) -> None:
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        (2, 5, 5, 2),
                        np.ones((2, 5, 5, 2), dtype=np.float32).flatten(),
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["axis"],
                    value=make_tensor("axis", TensorProto.INT64, (), (axis,)),
                ),
                make_node("DFT", ["input", "", "axis"], ["output"]),
            ],
            [],
        )
        with self.assertRaises(onnx.shape_inference.InferenceError):
            self._assert_inferred(
                graph,
                [
                    make_tensor_value_info("input", TensorProto.FLOAT, (2, 5, 5, 2)),
                    make_tensor_value_info("axis", TensorProto.INT64, ()),
                    make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 5, 2)),
                ],
                opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 20)],
            )

    @parameterized.expand(
        [
            ("real", (2, 5, 5, 1)),
            ("complex", (2, 5, 5, 2)),
        ]
    )
    def test_dft_dynamic_axis_opset20(self, _: str, shape: tuple[int, ...]) -> None:
        graph = self._make_graph(
            [("axis", TensorProto.INT64, ())],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        shape,
                        np.ones(shape, dtype=np.float32).flatten(),
                    ),
                ),
                make_node("DFT", ["input", "", "axis"], ["output"]),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, shape),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 5, 5, 2)),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 20)],
        )

    @parameterized.expand(
        [
            ("real", (2, 5, 5, 1)),
            ("complex", (2, 5, 5, 2)),
        ]
    )
    def test_dft_dynamic_axis_onesided_dft_length_opset20(
        self, _: str, shape: tuple[int, ...]
    ) -> None:
        graph = self._make_graph(
            [("axis", TensorProto.INT64, ())],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        shape,
                        np.ones(shape, dtype=np.float32).flatten(),
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["dft_length"],
                    value=make_tensor(
                        "dft_length",
                        TensorProto.INT64,
                        (),
                        np.array([42], dtype=np.int64),
                    ),
                ),
                make_node(
                    "DFT", ["input", "dft_length", "axis"], ["output"], onesided=1
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, shape),
                make_tensor_value_info("dft_length", TensorProto.INT64, ()),
                make_tensor_value_info(
                    "output", TensorProto.FLOAT, (None, None, None, 2)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 20)],
        )

    @parameterized.expand(
        [
            ("real", (2, 5, 5, 1)),
            ("complex", (2, 5, 5, 2)),
        ]
    )
    def test_dft_dynamic_axis_onesided_opset20(
        self, _: str, shape: tuple[int, ...]
    ) -> None:
        graph = self._make_graph(
            [("axis", TensorProto.INT64, ())],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        shape,
                        np.ones(shape, dtype=np.float32).flatten(),
                    ),
                ),
                make_node("DFT", ["input", "", "axis"], ["output"], onesided=1),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, shape),
                make_tensor_value_info(
                    "output", TensorProto.FLOAT, (None, None, None, 2)
                ),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 20)],
        )

    def test_dft_onesided_default_axis_opset17(self) -> None:
        # Opset 17 sets default axis to be 1.
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        (2, 5, 5, 2),
                        np.ones((2, 5, 5, 2), dtype=np.float32).flatten(),
                    ),
                ),
                make_node("DFT", ["input", ""], ["output"], onesided=1),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, (2, 5, 5, 2)),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 5, 2)),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 17)],
        )

    def test_dft_onesided_default_axis_opset20(self) -> None:
        # Opset 20 sets default axis to be -2.
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["input"],
                    value=make_tensor(
                        "input",
                        TensorProto.FLOAT,
                        (2, 5, 5, 2),
                        np.ones((2, 5, 5, 2), dtype=np.float32).flatten(),
                    ),
                ),
                make_node("DFT", ["input", "", ""], ["output"], onesided=1),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("input", TensorProto.FLOAT, (2, 5, 5, 2)),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 5, 3, 2)),
            ],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 20)],
        )

    def test_stft_reals(self):
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["signal"],
                    value=make_tensor(
                        "signal",
                        TensorProto.FLOAT,
                        (2, 10, 1),
                        (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3),
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["frame_step"],
                    value=make_tensor("frame_step", TensorProto.INT64, (), (2,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["window"],
                    value=make_tensor(
                        "window", TensorProto.INT64, (5,), (1, 2, 3, 4, 5)
                    ),
                ),
                make_node("STFT", ["signal", "frame_step", "window"], ["output"]),
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("signal", TensorProto.FLOAT, (2, 10, 1)),
                make_tensor_value_info("frame_step", TensorProto.INT64, ()),
                make_tensor_value_info("window", TensorProto.INT64, (5,)),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 5, 2)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["signal"],
                    value=make_tensor(
                        "signal",
                        TensorProto.FLOAT,
                        (2, 10, 1),
                        (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3),
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["frame_step"],
                    value=make_tensor("frame_step", TensorProto.INT64, (), (2,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["window"],
                    value=make_tensor(
                        "window", TensorProto.INT64, (5,), (1, 2, 3, 4, 5)
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["frame_length"],
                    value=make_tensor("frame_length", TensorProto.INT64, (), (5,)),
                ),
                make_node("STFT", ["signal", "frame_step", "window"], ["output"]),
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("signal", TensorProto.FLOAT, (2, 10, 1)),
                make_tensor_value_info("frame_step", TensorProto.INT64, ()),
                make_tensor_value_info("window", TensorProto.INT64, (5,)),
                make_tensor_value_info("frame_length", TensorProto.INT64, ()),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 5, 2)),
            ],
        )  # type: ignore

        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["signal"],
                    value=make_tensor(
                        "signal",
                        TensorProto.FLOAT,
                        (2, 10, 1),
                        (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3),
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["frame_step"],
                    value=make_tensor("frame_step", TensorProto.INT64, (), (2,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["frame_length"],
                    value=make_tensor("frame_length", TensorProto.INT64, (), (5,)),
                ),
                make_node(
                    "STFT", ["signal", "frame_step", "", "frame_length"], ["output"]
                ),
            ],
            [],
        )

        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("signal", TensorProto.FLOAT, (2, 10, 1)),
                make_tensor_value_info("frame_step", TensorProto.INT64, ()),
                make_tensor_value_info("frame_length", TensorProto.INT64, ()),
                make_tensor_value_info("output", TensorProto.FLOAT, (2, 3, 5, 2)),
            ],
        )  # type: ignore

    def test_melweightmatrix(self):
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["num_mel_bins"],
                    value=make_tensor("num_mel_bins", TensorProto.INT64, (), (10,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["dft_length"],
                    value=make_tensor("dft_length", TensorProto.INT64, (), (128,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["sample_rate"],
                    value=make_tensor("sample_rate", TensorProto.INT64, (), (10,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["lower_edge_hertz"],
                    value=make_tensor(
                        "lower_edge_hertz", TensorProto.FLOAT, (), (10.0,)
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["upper_edge_hertz"],
                    value=make_tensor(
                        "upper_edge_hertz", TensorProto.FLOAT, (), (100.0,)
                    ),
                ),
                make_node(
                    "MelWeightMatrix",
                    [
                        "num_mel_bins",
                        "dft_length",
                        "sample_rate",
                        "lower_edge_hertz",
                        "upper_edge_hertz",
                    ],
                    ["output"],
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("num_mel_bins", TensorProto.INT64, ()),
                make_tensor_value_info("dft_length", TensorProto.INT64, ()),
                make_tensor_value_info("sample_rate", TensorProto.INT64, ()),
                make_tensor_value_info("lower_edge_hertz", TensorProto.FLOAT, ()),
                make_tensor_value_info("upper_edge_hertz", TensorProto.FLOAT, ()),
                make_tensor_value_info("output", TensorProto.FLOAT, (65, 10)),
            ],
        )  # type: ignore

    def test_melweightmatrix_with_output_datatype(self):
        graph = self._make_graph(
            [],
            [
                make_node(
                    "Constant",
                    [],
                    ["num_mel_bins"],
                    value=make_tensor("num_mel_bins", TensorProto.INT64, (), (10,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["dft_length"],
                    value=make_tensor("dft_length", TensorProto.INT64, (), (128,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["sample_rate"],
                    value=make_tensor("sample_rate", TensorProto.INT64, (), (10,)),
                ),
                make_node(
                    "Constant",
                    [],
                    ["lower_edge_hertz"],
                    value=make_tensor(
                        "lower_edge_hertz", TensorProto.FLOAT, (), (10.0,)
                    ),
                ),
                make_node(
                    "Constant",
                    [],
                    ["upper_edge_hertz"],
                    value=make_tensor(
                        "upper_edge_hertz", TensorProto.FLOAT, (), (100.0,)
                    ),
                ),
                make_node(
                    "MelWeightMatrix",
                    [
                        "num_mel_bins",
                        "dft_length",
                        "sample_rate",
                        "lower_edge_hertz",
                        "upper_edge_hertz",
                    ],
                    ["output"],
                    output_datatype=TensorProto.DOUBLE,
                ),
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("num_mel_bins", TensorProto.INT64, ()),
                make_tensor_value_info("dft_length", TensorProto.INT64, ()),
                make_tensor_value_info("sample_rate", TensorProto.INT64, ()),
                make_tensor_value_info("lower_edge_hertz", TensorProto.FLOAT, ()),
                make_tensor_value_info("upper_edge_hertz", TensorProto.FLOAT, ()),
                make_tensor_value_info("output", TensorProto.DOUBLE, (65, 10)),
            ],
        )  # type: ignore

    def test_center_crop_pad_hwc_crop(self):
        graph = self._make_graph(
            [
                ("input_data", TensorProto.FLOAT, (20, 10, 3)),
                ("shape", TensorProto.INT64, (2,)),
            ],
            [make_node("CenterCropPad", ["input_data", "shape"], ["y"], axes=[0, 1])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (10, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (10, 8, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)],
        )

    def test_center_crop_pad_chw_crop(self):
        graph = self._make_graph(
            [
                ("input_data", TensorProto.FLOAT, (3, 20, 10)),
                ("shape", TensorProto.INT64, (2,)),
            ],
            [make_node("CenterCropPad", ["input_data", "shape"], ["y"], axes=[1, 2])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (10, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (3, 10, 8))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)],
        )

    def test_center_crop_pad_hwc_croppad(self):
        graph = self._make_graph(
            [
                ("input_data", TensorProto.FLOAT, (10, 10, 3)),
                ("shape", TensorProto.INT64, (2,)),
            ],
            [make_node("CenterCropPad", ["input_data", "shape"], ["y"], axes=[0, 1])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (20, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (20, 8, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)],
        )

    def test_center_crop_pad_chw_croppad(self):
        graph = self._make_graph(
            [
                ("input_data", TensorProto.FLOAT, (3, 10, 10)),
                ("shape", TensorProto.INT64, (2,)),
            ],
            [make_node("CenterCropPad", ["input_data", "shape"], ["y"], axes=[1, 2])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (20, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (3, 20, 8))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)],
        )

    def test_center_crop_pad_without_input_shape(self):
        graph = self._make_graph(
            [
                ("input_data", TensorProto.FLOAT, (3, 2)),
                ("shape", TensorProto.INT64, (2,)),
            ],
            [make_node("CenterCropPad", ["input_data", "shape"], ["y"])],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, None)],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)],
        )

    def test_center_crop_pad_with_input_shape_containing_dim_params(
        self,
    ):
        graph = self._make_graph(
            [
                ("input_data", TensorProto.FLOAT, (20, "W", 3)),
                ("shape", TensorProto.INT64, (2,)),
            ],
            [make_node("CenterCropPad", ["input_data", "shape"], ["y"], axes=[0, 1])],
            [],
            initializer=[make_tensor("shape", TensorProto.INT64, (2,), (10, 8))],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (10, 8, 3))],
            opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)],
        )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_category_mapper(self) -> None:
        cat = make_node(
            "CategoryMapper",
            ["x"],
            ["y"],
            domain=ONNX_ML_DOMAIN,
        )
        graph_int = self._make_graph(
            [("x", TensorProto.INT64, (30, 4, 5))],
            [cat],
            [],
        )
        self._assert_inferred(
            graph_int,
            [make_tensor_value_info("y", TensorProto.STRING, (30, 4, 5))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )
        graph_str = self._make_graph(
            [("x", TensorProto.STRING, (30, 5, 4))],
            [cat],
            [],
        )
        self._assert_inferred(
            graph_str,
            [make_tensor_value_info("y", TensorProto.INT64, (30, 5, 4))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_tree_ensemble_regressor(self) -> None:
        tree = make_node(
            "TreeEnsembleRegressor",
            ["x"],
            ["y"],
            domain=ONNX_ML_DOMAIN,
            n_targets=5,
        )
        graph = self._make_graph(
            [("x", TensorProto.DOUBLE, (30, 3))],
            [tree],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.FLOAT, (30, 5))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 3),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @parameterized.expand([TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16])
    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_tree_ensemble(self, dtype) -> None:
        interior_nodes = 5
        leaves = 9
        tree = make_node(
            "TreeEnsemble",
            ["x"],
            ["y"],
            domain=ONNX_ML_DOMAIN,
            n_targets=5,
            nodes_featureids=[0] * interior_nodes,
            nodes_splits=make_tensor(
                "nodes_splits",
                dtype,
                (interior_nodes,),
                list(range(interior_nodes)),
            ),
            nodes_modes=make_tensor(
                "nodes_modes",
                TensorProto.UINT8,
                (interior_nodes,),
                [0] * interior_nodes,
            ),
            nodes_truenodeids=[0] * interior_nodes,
            nodes_falsenodeids=[0] * interior_nodes,
            nodes_trueleafs=[0] * interior_nodes,
            nodes_falseleafs=[0] * interior_nodes,
            membership_values=make_tensor(
                "membership_values",
                dtype,
                (7,),
                [0.0, 0.1, 0.2, np.nan, 0.4, 0.5, 1.0],
            ),
            leaf_targetids=[0] * leaves,
            leaf_weights=make_tensor("leaf_weights", dtype, (leaves,), [1] * leaves),
            tree_roots=[0],
        )

        graph = self._make_graph(
            [("x", dtype, ("Batch Size", "Features"))],
            [tree],
            [],
        )

        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", dtype, ("Batch Size", 5))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 5),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @parameterized.expand(
        [
            {
                "nodes_truenodeids": [0] * 6,
                "leaf_weights": make_tensor(
                    "leaf_weights", TensorProto.DOUBLE, (9,), [1] * 9
                ),
                "nodes_splits": make_tensor(
                    "nodes_splits", TensorProto.DOUBLE, (5,), [1] * 5
                ),
            },
            {
                "nodes_truenodeids": [0] * 5,
                "leaf_weights": make_tensor(
                    "leaf_weights", TensorProto.FLOAT, (9,), [1] * 9
                ),
                "nodes_splits": make_tensor(
                    "nodes_splits", TensorProto.DOUBLE, (5,), [1] * 5
                ),
            },
            {
                "nodes_truenodeids": [0] * 5,
                "leaf_weights": make_tensor(
                    "leaf_weights", TensorProto.DOUBLE, (18,), [1] * 18
                ),
                "nodes_splits": make_tensor(
                    "nodes_splits", TensorProto.DOUBLE, (5,), [1] * 5
                ),
            },
            {
                "nodes_truenodeids": [0] * 5,
                "leaf_weights": make_tensor(
                    "leaf_weights", TensorProto.DOUBLE, (9,), [1] * 9
                ),
                "nodes_splits": make_tensor(
                    "nodes_splits", TensorProto.FLOAT, (5,), [1] * 5
                ),
            },
        ]
    )
    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_tree_ensemble_fails_if_invalid_attributes(
        self,
        nodes_truenodeids,
        leaf_weights,
        nodes_splits,
    ) -> None:
        interior_nodes = 5
        leaves = 9
        tree = make_node(
            "TreeEnsemble",
            ["x"],
            ["y"],
            domain=ONNX_ML_DOMAIN,
            n_targets=5,
            nodes_featureids=[0] * interior_nodes,
            nodes_splits=nodes_splits,
            nodes_modes=make_tensor(
                "nodes_modes",
                TensorProto.UINT8,
                (interior_nodes,),
                [0] * interior_nodes,
            ),
            nodes_truenodeids=nodes_truenodeids,
            nodes_falsenodeids=[0] * interior_nodes,
            nodes_trueleafs=[0] * interior_nodes,
            nodes_falseleafs=[0] * interior_nodes,
            leaf_targetids=[0] * leaves,
            leaf_weights=leaf_weights,
            tree_roots=[0],
        )

        graph = self._make_graph(
            [("x", TensorProto.DOUBLE, ("Batch Size", "Features"))],
            [tree],
            [],
        )
        self.assertRaises(onnx.shape_inference.InferenceError, self._inferred, graph)

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_tree_ensemble_classifier(self) -> None:
        tree = make_node(
            "TreeEnsembleClassifier",
            ["x"],
            ["y", "z"],
            classlabels_int64s=[0, 1, 2, 3, 4],
            domain=ONNX_ML_DOMAIN,
        )
        graph = self._make_graph(
            [("x", TensorProto.DOUBLE, (30, 3))],
            [tree],
            [],
        )
        self._assert_inferred(
            graph,
            [
                make_tensor_value_info("y", TensorProto.INT64, (30,)),
                make_tensor_value_info("z", TensorProto.FLOAT, (30, 5)),
            ],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 3),
                make_opsetid(ONNX_DOMAIN, 11),
            ],
        )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_array_feature_extractor(self) -> None:
        node = make_node(
            "ArrayFeatureExtractor",
            ["x", "y"],
            ["z"],
            domain=ONNX_ML_DOMAIN,
        )
        for axes_shape, expected in [
            ((2,), 2),
            ((), "unk__0"),
            (("N",), "N"),
        ]:
            graph = self._make_graph(
                [
                    ("x", TensorProto.INT64, (3, 4, 5)),
                    ("y", TensorProto.INT64, axes_shape),
                ],
                [node],
                [],
            )
            self._assert_inferred(
                graph,
                [make_tensor_value_info("z", TensorProto.INT64, (3, 4, expected))],  # type: ignore
                opset_imports=[
                    make_opsetid(ONNX_ML_DOMAIN, 3),
                    make_opsetid(ONNX_DOMAIN, 18),
                ],
            )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_binarizer(self) -> None:
        node = make_node(
            "Binarizer",
            ["x"],
            ["y"],
            domain=ONNX_ML_DOMAIN,
        )
        graph = self._make_graph(
            [
                ("x", TensorProto.INT64, (3, 4, 5)),
            ],
            [node],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("y", TensorProto.INT64, (3, 4, 5))],  # type: ignore
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 3),
                make_opsetid(ONNX_DOMAIN, 18),
            ],
        )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_one_hot_encoder(self) -> None:
        graph = self._make_graph(
            [("input", TensorProto.INT64, (2, "N", 3))],
            [
                make_node(
                    "OneHotEncoder",
                    ["input"],
                    ["output"],
                    cats_int64s=[1, 2, 3, 4],
                    domain="ai.onnx.ml",
                )
            ],
            [],
        )
        self._assert_inferred(
            graph,
            [make_tensor_value_info("output", TensorProto.FLOAT, (2, "N", 3, 4))],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 18),
            ],
        )

    @unittest.skipUnless(ONNX_ML, "ONNX_ML required to test ai.onnx.ml operators")
    def test_zip_map(self) -> None:
        params = (
            ({"classlabels_int64s": [1, 2, 3]}, onnx.TensorProto.INT64),
            ({"classlabels_strings": ["a", "b", "c"]}, onnx.TensorProto.STRING),
        )
        for attrs, input_type in params:
            with self.subTest(attrs=attrs, input_type=input_type):
                self.zip_map_test_case(attrs, input_type)

    def zip_map_test_case(self, attrs, input_type) -> None:
        graph = self._make_graph(
            [("input", TensorProto.FLOAT, ("N", 3))],
            [
                make_node(
                    "ZipMap",
                    ["input"],
                    ["output"],
                    **attrs,
                    domain="ai.onnx.ml",
                )
            ],
            [],
        )
        typ = onnx.helper.make_map_type_proto(
            input_type, onnx.helper.make_tensor_type_proto(TensorProto.FLOAT, ())
        )
        self._assert_inferred(
            graph,
            [
                onnx.helper.make_value_info(
                    "output", onnx.helper.make_sequence_type_proto(typ)
                )
            ],
            opset_imports=[
                make_opsetid(ONNX_ML_DOMAIN, 1),
                make_opsetid(ONNX_DOMAIN, 18),
            ],
        )

    def test_compress_without_axis(self) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.INT64, (2, "N", 3, 4)),
                ("condition", TensorProto.BOOL, (None,)),
            ],
            [make_node("Compress", ["input", "condition"], ["output"])],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("output", TensorProto.INT64, (None,))])  # type: ignore

    def test_compress_with_axis(self) -> None:
        graph = self._make_graph(
            [
                ("input", TensorProto.INT64, (2, "N", 3, 4)),
                ("condition", TensorProto.BOOL, (None,)),
            ],
            [make_node("Compress", ["input", "condition"], ["output"], axis=-1)],
            [],
        )
        self._assert_inferred(graph, [make_tensor_value_info("output", TensorProto.INT64, (2, "N", 3, None))])  # type: ignore

    def test_check_type_when_schema_has_empty_io(self):
        input = """
            <
                ir_version: 7,
                opset_import: ["" : 1]
            >
            agraph (X, Y) => (Z)
            {
                Z = CustomOp(X, Y)
            }
           """
        model = onnx.parser.parse_model(input)

        op_schema = defs.OpSchema(
            "CustomOp",
            "",
            1,
            inputs=[],
            outputs=[],
        )
        onnx.defs.register_schema(op_schema)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.shape_inference.infer_shapes(model, True)
        onnx.defs.deregister_schema(
            op_schema.name, op_schema.since_version, op_schema.domain
        )

    def test_issue_layer_normalization_6187(self):
        modeltxt = """
        <
        ir_version: 10,
        opset_import: ["" : 17]
        >
        graph (float in0, float[2,7,8,1,3] in1, float[3,7] in2) => () {
        out0, out1, out2 = LayerNormalization <epsilon: float = -841.058, stash_type: int = -940> (in0, in1, in2)
        }
        """
        model = onnx.parser.parse_model(modeltxt)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.checker.check_model(model, full_check=True)
            onnx.shape_inference.infer_shapes(model)

    def test_issue_conv_6180(self):
        modeltxt = """
        <
        ir_version: 9,
        opset_import: ["" : 11]
        >
        graph (float[7,6,1,5] in0, float in1, float[7,2,3,2,1] in2) => () {
        out0 = Conv <auto_pad = "NOTSET", group = 1> (in0, in1, in2)
        }
        """
        model = onnx.parser.parse_model(modeltxt)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.checker.check_model(model, full_check=True)
            onnx.shape_inference.infer_shapes(model)

    def test_issue_gemm_6185(self):
        modeltxt = """
        <
        ir_version: 10,
        opset_import: ["" : 6]
        >
        graph (double[2,1] in0, double in1, double[2] in2) => () {
        out0 = Gemm <alpha: float = 1, beta: float = -693.752, broadcast: int = -436, transB: int = 823> (in0, in1, in2)
        }
        """
        model = onnx.parser.parse_model(modeltxt)
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(model, full_check=True)
            onnx.shape_inference.infer_shapes(model)

    def test_issue_stft_6186(self):
        modeltxt = """
        <
        ir_version: 10,
        opset_import: ["" : 17]
        >
        graph (float16[3] in0, int32[2] in1, float16[7,8,8,8] in2, int32[8,1,7,2] in3) => () {
        out0 = STFT (in0, in1, in2, in3)
        }
        """
        model = onnx.parser.parse_model(modeltxt)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.checker.check_model(model, full_check=True)
            onnx.shape_inference.infer_shapes(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
