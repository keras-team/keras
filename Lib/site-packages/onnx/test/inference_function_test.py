# SPDX-License-Identifier: Apache-2.0

# Copyright (c) ONNX Project Contributors
from __future__ import annotations

import unittest

import numpy as np

import onnx
from onnx import TensorProto, TypeProto
from onnx.checker import ValidationError
from onnx.defs import OpSchema, get_all_schemas_with_history, get_schema
from onnx.helper import (
    make_graph,
    make_node,
    make_opsetid,
    make_tensor_type_proto,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnx.shape_inference import InferenceError, infer_node_outputs

ADD_SCHEMA = max(
    (s for s in get_all_schemas_with_history() if s.name == "Add" and s.domain == ""),
    key=lambda s: s.since_version,
)
RESHAPE_SCHEMA = max(
    (
        s
        for s in get_all_schemas_with_history()
        if s.name == "Reshape" and s.domain == ""
    ),
    key=lambda s: s.since_version,
)
CLIP_SCHEMA = max(
    (s for s in get_all_schemas_with_history() if s.name == "Clip" and s.domain == ""),
    key=lambda s: s.since_version,
)


def _to_tensor_types(
    tensor_types: dict[str, tuple[int, tuple[int | str | None, ...]]]
) -> dict[str, TypeProto]:
    return {key: make_tensor_type_proto(*value) for key, value in tensor_types.items()}


def _run_case(
    schema: OpSchema,
    input_names: list[str],
    output_names: list[str],
    input_types: dict[str, TypeProto],
    input_data: dict[str, np.ndarray] | None = None,
) -> dict[str, TypeProto]:
    if input_data is None:
        input_data = {}
    return infer_node_outputs(
        schema,
        make_node(schema.name, input_names, output_names, domain=schema.domain),
        input_types,
        {key: from_array(arr) for key, arr in input_data.items()},
    )


class TestInferenceFunctionCall(unittest.TestCase):
    def test_add_inference(self) -> None:
        cases = [
            (
                {"A": (TensorProto.FLOAT, ()), "B": (TensorProto.FLOAT, ())},
                {"C": (TensorProto.FLOAT, ())},
            ),
            (
                {
                    "A": (TensorProto.FLOAT, (None, 2)),
                    "B": (TensorProto.FLOAT, (2,)),
                },
                {"C": (TensorProto.FLOAT, (None, 2))},
            ),
            (
                {
                    "A": (TensorProto.FLOAT, (None, 2)),
                    "B": (TensorProto.FLOAT, (1, 2)),
                },
                {"C": (TensorProto.FLOAT, (None, 2))},
            ),
            (
                {
                    "A": (TensorProto.DOUBLE, ("n", "m")),
                    "B": (TensorProto.DOUBLE, (1, "n", "m")),
                },
                {"C": (TensorProto.DOUBLE, (1, "n", "m"))},
            ),
            (
                {
                    "A": (TensorProto.FLOAT, ("x", 2)),
                    "B": (TensorProto.FLOAT, ("y", 2)),
                },
                {"C": (TensorProto.FLOAT, (None, 2))},
            ),
        ]
        for ins, outs in cases:
            assert _run_case(ADD_SCHEMA, ["A", "B"], ["C"], _to_tensor_types(ins)) == _to_tensor_types(outs)  # type: ignore

    def test_clip_inference_with_optional_input(self) -> None:
        # Test case where the second input is optional
        input_names = ["X", "", "max"]
        output_names = ["Y"]
        input_types = _to_tensor_types(
            {"X": (TensorProto.FLOAT, (3, 4)), "max": (TensorProto.FLOAT, ())}
        )
        expected_output_types = _to_tensor_types({"Y": (TensorProto.FLOAT, (3, 4))})
        assert (
            _run_case(CLIP_SCHEMA, input_names, output_names, input_types)
            == expected_output_types
        )

    def test_add_inference_raises_errors(self) -> None:
        with self.assertRaises(ValidationError):
            _run_case(
                ADD_SCHEMA,
                ["A"],
                ["C"],
                _to_tensor_types({"A": (TensorProto.FLOAT, (3, 4))}),
            )
        with self.assertRaises(ValidationError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types({"A": (TensorProto.FLOAT, (3, 4)), "B": (2, (3, 4))}),
            )
        with self.assertRaises(InferenceError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types(
                    {
                        "A": (TensorProto.FLOAT, (2, 4)),
                        "B": (TensorProto.FLOAT, (3, 4)),
                    }
                ),
            )
        with self.assertRaises(KeyError):
            _run_case(
                ADD_SCHEMA,
                ["A", "B"],
                ["C"],
                _to_tensor_types({"A": (TensorProto.FLOAT, (3, 4))}),
            )

    def test_reshape_inference(self) -> None:
        assert _run_case(
            RESHAPE_SCHEMA,
            ["x", "t"],
            ["y"],
            _to_tensor_types(
                {
                    "x": (TensorProto.FLOAT, (5, 4)),
                    "t": (TensorProto.INT64, (3,)),
                }
            ),
            {"t": np.array([2, 2, 5], dtype=np.int64)},
        ) == _to_tensor_types({"y": (TensorProto.FLOAT, (2, 2, 5))})

    def test_scan_inference_with_subgraph(self) -> None:
        seq_len = "sequence"
        input_size = 2
        loop_state_size = 3

        input_value_infos = [
            make_tensor_value_info("loop_state_in", TensorProto.UNDEFINED, None),
            make_tensor_value_info("input", TensorProto.UNDEFINED, None),
            make_tensor_value_info("outer", TensorProto.UNDEFINED, None),
        ]
        output_value_infos = [
            make_tensor_value_info("loop_state_out", TensorProto.UNDEFINED, None),
            make_tensor_value_info("output", TensorProto.FLOAT, (seq_len, input_size)),
        ]

        subgraph = make_graph(
            [
                make_node("Identity", ["loop_state_in"], ["loop_state_out"]),
                make_node("Add", ["input", "outer"], ["output"]),
            ],
            "subgraph",
            input_value_infos,
            output_value_infos,
        )

        assert infer_node_outputs(
            get_schema("Scan", 9),
            make_node(
                "Scan",
                ["loop_state_orig", "scan_input", "scan_outer"],
                ["loop_state_final", "scan_output"],
                num_scan_inputs=1,
                body=subgraph,
            ),
            _to_tensor_types(
                {
                    "loop_state_orig": (TensorProto.FLOAT, (loop_state_size,)),
                    "scan_input": (TensorProto.FLOAT, (seq_len, input_size)),
                    "scan_outer": (TensorProto.FLOAT, (input_size,)),
                }
            ),
            # Same as default value in Scan-9
            opset_imports=[make_opsetid("", 9)],
            ir_version=4,
        ) == _to_tensor_types(
            {
                "loop_state_final": (TensorProto.FLOAT, (loop_state_size,)),
                "scan_output": (TensorProto.FLOAT, (seq_len, input_size)),
            }
        )

    def test_inference_with_conflow(self) -> None:
        model_script = """
        <
            ir_version: 8,
            opset_import: ["" : 18, "onnxscript.atenlib" : 1],
            producer_name: "pytorch",
            producer_version: "2.1.0"
        >
        torch_jit (float input_0) => (float reault, int64 index)
        {
            reault, index = onnxscript.atenlib.aten_min_dim <dim = 0, keepdim = 1> (input_0)
        }
        <
            domain: "onnxscript.atenlib",
            opset_import: ["" : 18]
        >
        aten_min_dim <dim>(self) => (result_7, indices_6)
        {
            tmp = Shape (self)
            tmp_0 = Size (tmp)
            tmp_1 = Constant <value = int64 tmp_1 {0}> ()
            tmp_1_cast = CastLike (tmp_1, tmp_0)
            tmp_2 = Equal (tmp_0, tmp_1_cast)
            cond = Not (tmp_2)
            indices_6, result_7 = If (cond) <
                then_branch = thenGraph_4 () => ( indices,  result) {
                    dim = Constant <value_int: int = @dim> ()
                    tmp_3 = Constant <value_ints = [-1]> ()
                    dims = Reshape (dim, tmp_3)
                    result = ReduceMin <keepdims: int = @keepdim> (self, dims)
                    indices = ArgMin <axis: int = @dim, keepdims: int = @keepdim> (self)
                }, else_branch = elseGraph_4 () => ( indices_4,  result_5) {
                    indices_4 = Constant <value_int = 0> ()
                    result_5 = Identity (self)
                }
            >
        }
        """
        model = onnx.parser.parse_model(model_script)
        onnx.shape_inference.infer_shapes(model, strict_mode=False)
        with self.assertRaises(onnx.shape_inference.InferenceError):
            onnx.shape_inference.infer_shapes(model, strict_mode=True)

    def test_inference_with_attribute(self) -> None:
        model_script = """
        <
            ir_version: 8,
            opset_import: ["" : 18, "custom" : 1],
            producer_name: "",
            producer_version: "1.0"
        >
        MeanVarianceNormalization (float[N] x) => (float[M] y)
        {
            y = custom.custom_mvn <axes = [0]> (x)
        }
        <
            domain: "custom",
            opset_import: ["" : 18]
        >
        custom_mvn <axes>(X) => (Y)
        {
          Exponent = Constant <value = float {2.0}>()
          Epsilon = Constant <value = float {1e-9}>()
          axes = Constant <value_ints: ints = @axes>()
          X_RM = ReduceMean (X, axes)
          EX_squared = Pow (X_RM, Exponent)
          X_squared = Pow (X, Exponent)
          E_Xsquared = ReduceMean (X_squared, axes)
          Variance = Sub (E_Xsquared, EX_squared)
          STD = Sqrt (Variance)
          X_variance = Sub (X, X_RM)
          Processed_STD = Add (STD, Epsilon)
          Y = Div (X_variance, Processed_STD)
        }
        """
        model = onnx.parser.parse_model(model_script)
        # onnx.shape_inference.infer_shapes(model, strict_mode=False)
        onnx.shape_inference.infer_shapes(model, strict_mode=True)


if __name__ == "__main__":
    unittest.main()
