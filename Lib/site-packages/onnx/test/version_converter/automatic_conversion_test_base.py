# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import string
import unittest
from typing import Any, List, Sequence, cast

import onnx
from onnx import TensorProto, ValueInfoProto, helper, shape_inference, version_converter

LATEST_OPSET = onnx.defs.onnx_opset_version()


class TestAutomaticConversion(unittest.TestCase):
    def _test_model_conversion(
        self, to_opset: int, model: str | onnx.ModelProto
    ) -> None:
        if isinstance(model, str):
            model = onnx.parser.parse_model(model)
        onnx.checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        converted = version_converter.convert_version(model, to_opset)
        onnx.checker.check_model(converted)
        shape_inference.infer_shapes(converted, strict_mode=True)

    def _test_model_conversion_fails(
        self, to_opset: int, model: str | onnx.ModelProto
    ) -> None:
        if isinstance(model, str):
            model = onnx.parser.parse_model(model)
        onnx.checker.check_model(model)
        shape_inference.infer_shapes(model, strict_mode=True)

        with self.assertRaises(RuntimeError):
            version_converter.convert_version(model, to_opset)

    def _test_op_conversion(
        self,
        op: str,
        from_opset: int,
        input_shapes: Sequence[Sequence[int | None] | str] = ((3, 4, 5),),
        output_shapes: Sequence[Sequence[int | None]] = ((3, 4, 5),),
        input_types: Sequence[Any] | None = None,
        output_types: Sequence[Any] | None = None,
        initializer: Sequence[Any] = (),
        attrs: dict[str, Any] | None = None,
        seq_inputs: Sequence[int] = (),
        seq_outputs: Sequence[int] = (),
        optional_inputs: Sequence[int] = (),
        optional_outputs: Sequence[int] = (),
        is_upgrade: bool = True,
    ) -> None:
        """Test conversion.

        Args:
            op: A string representing the name of the operator to test.
            from_opset: An integer representing the lowest opset version to convert.
            input_shapes: A sequence of tuples or strings representing the shapes of the input tensors.
                The default value is ((3, 4, 5),).
            output_shapes: A sequence of tuples representing the shapes of the output tensors.
                The default value is ((3, 4, 5),).
            input_types: An optional sequence of types representing the data types of the input tensors.
            output_types: An optional sequence of types representing the data types of the output tensors.
            initializer: A sequence of values representing the initial values of the input tensors.
            attrs: An optional dictionary of attributes for the operator.
            seq_inputs: A sequence of integers representing the indices of the input tensors that are sequences.
            seq_outputs: A sequence of integers representing the indices of the output tensors that are sequences.
            optional_inputs: A sequence of integers representing the indices of the input tensors that are optional.
            optional_outputs: A sequence of integers representing the indices of the output tensors that are optional.
            is_upgrade: A boolean value indicating whether to run the version converter from from_opset to
                the most recent opset version (True) or from the most recent opset version to from_opset (False).
                The default value is True. In both cases, runs checker and shape inference on the final model.
        """
        if attrs is None:
            attrs = {}

        n_inputs = len(input_shapes)
        letters = list(string.ascii_lowercase)[:n_inputs]
        input_names = [
            letter if shape != "" else ""
            for (letter, shape) in zip(letters, input_shapes)
        ]
        if input_types is None:
            input_types = [TensorProto.FLOAT] * n_inputs
        is_sequence = [0 if id not in seq_inputs else 1 for id in range(n_inputs)]
        is_optional = [0 if id not in optional_inputs else 1 for id in range(n_inputs)]
        # turn empty strings into [0] to ease type analysis, even though those entries
        # will be ignored
        input_shapes_cast = cast(
            List[List[int]],
            [[0] if isinstance(shape, str) else shape for shape in input_shapes],
        )
        inputs: list[ValueInfoProto] = []
        for name, ttype, shape, is_seq, is_opt in zip(
            input_names, input_types, input_shapes_cast, is_sequence, is_optional
        ):
            if name != "":
                if is_seq:
                    inputs += [
                        helper.make_tensor_sequence_value_info(name, ttype, shape)
                    ]
                elif is_opt:
                    type_proto = helper.make_tensor_type_proto(ttype, shape)
                    optional_type_proto = helper.make_optional_type_proto(type_proto)
                    inputs += [helper.make_value_info(name, optional_type_proto)]
                else:
                    inputs += [helper.make_tensor_value_info(name, ttype, shape)]

        n_outputs = len(output_shapes)
        output_names = list(string.ascii_lowercase)[n_inputs : n_inputs + n_outputs]
        if output_types is None:
            output_types = [TensorProto.FLOAT] * n_outputs
        is_sequence = [0 if id not in seq_outputs else 1 for id in range(n_outputs)]
        is_optional = [
            0 if id not in optional_outputs else 1 for id in range(n_outputs)
        ]
        output_shapes_cast = cast(
            List[List[int]],
            [[0] if isinstance(shape, str) else shape for shape in output_shapes],
        )
        outputs: list[ValueInfoProto] = []
        for name, ttype, shape, is_seq, is_opt in zip(
            output_names, output_types, output_shapes_cast, is_sequence, is_optional
        ):
            if is_seq:
                outputs += [helper.make_tensor_sequence_value_info(name, ttype, shape)]
            elif is_opt:
                type_proto = helper.make_tensor_type_proto(ttype, shape)
                optional_type_proto = helper.make_optional_type_proto(type_proto)
                outputs += [helper.make_value_info(name, optional_type_proto)]
            else:
                outputs += [helper.make_tensor_value_info(name, ttype, shape)]

        node = helper.make_node(op, input_names, output_names, **attrs)
        graph = helper.make_graph([node], op, inputs, outputs, initializer)
        start_opset = from_opset if is_upgrade else LATEST_OPSET
        end_opset = LATEST_OPSET if is_upgrade else from_opset
        original = helper.make_model(
            graph,
            producer_name="test",
            opset_imports=[helper.make_opsetid("", start_opset)],
        )
        self._test_model_conversion(end_opset, original)
