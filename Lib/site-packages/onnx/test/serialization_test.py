# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
import unittest

import onnx

_TEST_MODEL = """\
<
    ir_version: 8,
    opset_import: ["" : 17, "local" : 1]
>
agraph (float[N] X) => (float[N] Y) {
    Y = local.foo (X)
}

<opset_import: ["" : 17, "local" : 1], domain: "local">
foo (x) => (y) {
    temp = Add(x, x)
    y = local.bar(temp)
}

<opset_import: ["" : 17], domain: "local">
bar (x) => (y) {
    y = Mul (x, x)
}"""


class _OnnxTestTextualSerializer(onnx.serialization.ProtoSerializer):
    """Serialize and deserialize the ONNX textual representation."""

    supported_format = "onnxtext"
    file_extensions = frozenset({".onnxtext"})

    def serialize_proto(self, proto) -> bytes:
        text = onnx.printer.to_text(proto)
        return text.encode("utf-8")

    def deserialize_proto(self, serialized: bytes, proto):
        text = serialized.decode("utf-8")
        if isinstance(proto, onnx.ModelProto):
            return onnx.parser.parse_model(text)
        if isinstance(proto, onnx.GraphProto):
            return onnx.parser.parse_graph(text)
        if isinstance(proto, onnx.FunctionProto):
            return onnx.parser.parse_function(text)
        if isinstance(proto, onnx.NodeProto):
            return onnx.parser.parse_node(text)
        raise ValueError(f"Unsupported proto type: {type(proto)}")


class TestRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.serializer = _OnnxTestTextualSerializer()
        onnx.serialization.registry.register(self.serializer)

    def test_get_returns_the_registered_instance(self) -> None:
        serializer = onnx.serialization.registry.get("onnxtext")
        self.assertIs(serializer, self.serializer)

    def test_get_raises_for_unsupported_format(self) -> None:
        with self.assertRaises(ValueError):
            onnx.serialization.registry.get("unsupported")

    def test_onnx_save_load_model_uses_the_custom_serializer(self) -> None:
        model = onnx.parser.parse_model(_TEST_MODEL)
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            onnx.save_model(model, model_path, format="onnxtext")

            # Check the file content
            with open(model_path, encoding="utf-8") as f:
                content = f.read()
                self.assertEqual(content, onnx.printer.to_text(model))

            loaded_model = onnx.load_model(model_path, format="onnxtext")

            self.assertEqual(
                model.SerializeToString(deterministic=True),
                loaded_model.SerializeToString(deterministic=True),
            )


class TestCustomSerializer(unittest.TestCase):
    def test_serialize_deserialize_model(self) -> None:
        serializer = _OnnxTestTextualSerializer()
        model = onnx.parser.parse_model(_TEST_MODEL)
        serialized = serializer.serialize_proto(model)
        deserialized = serializer.deserialize_proto(serialized, onnx.ModelProto())
        self.assertEqual(
            model.SerializeToString(deterministic=True),
            deserialized.SerializeToString(deterministic=True),
        )
