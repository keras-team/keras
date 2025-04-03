# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os
import pathlib
import tempfile
import unittest

import google.protobuf.message
import google.protobuf.text_format
import parameterized

import onnx
from onnx import serialization


def _simple_model() -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.ir_version = onnx.IR_VERSION
    model.producer_name = "onnx-test"
    model.graph.name = "test"
    return model


def _simple_tensor() -> onnx.TensorProto:
    tensor = onnx.helper.make_tensor(
        name="test-tensor",
        data_type=onnx.TensorProto.FLOAT,
        dims=(2, 3, 4),
        vals=[x + 0.5 for x in range(24)],
    )
    return tensor


@parameterized.parameterized_class(
    [
        {"format": "protobuf"},
        {"format": "textproto"},
        {"format": "json"},
        {"format": "onnxtxt"},
    ]
)
class TestIO(unittest.TestCase):
    format: str

    def test_load_model_when_input_is_bytes(self) -> None:
        proto = _simple_model()
        proto_string = serialization.registry.get(self.format).serialize_proto(proto)
        loaded_proto = onnx.load_model_from_string(proto_string, format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_has_read_function(self) -> None:
        proto = _simple_model()
        # When the proto is a bytes representation provided to `save_model`,
        # it should always be a serialized binary protobuf representation. Aka. format="protobuf"
        # The saved file format is specified by the `format` argument.
        proto_string = serialization.registry.get("protobuf").serialize_proto(proto)
        f = io.BytesIO()
        onnx.save_model(proto_string, f, format=self.format)
        loaded_proto = onnx.load_model(io.BytesIO(f.getvalue()), format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_file_name(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.onnx")
            onnx.save_model(proto, model_path, format=self.format)
            loaded_proto = onnx.load_model(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_model_when_input_is_pathlike(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = pathlib.Path(temp_dir, "model.onnx")
            onnx.save_model(proto, model_path, format=self.format)
            loaded_proto = onnx.load_model(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)


@parameterized.parameterized_class(
    [
        {"format": "protobuf"},
        {"format": "textproto"},
        {"format": "json"},
        # The onnxtxt format does not support saving/loading tensors yet
    ]
)
class TestIOTensor(unittest.TestCase):
    """Test loading and saving of TensorProto."""

    format: str

    def test_load_tensor_when_input_is_bytes(self) -> None:
        proto = _simple_tensor()
        proto_string = serialization.registry.get(self.format).serialize_proto(proto)
        loaded_proto = onnx.load_tensor_from_string(proto_string, format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_has_read_function(self) -> None:
        # Test if input has a read function
        proto = _simple_tensor()
        f = io.BytesIO()
        onnx.save_tensor(proto, f, format=self.format)
        loaded_proto = onnx.load_tensor(io.BytesIO(f.getvalue()), format=self.format)
        self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_file_name(self) -> None:
        # Test if input is a file name
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.onnx")
            onnx.save_tensor(proto, model_path, format=self.format)
            loaded_proto = onnx.load_tensor(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)

    def test_save_and_load_tensor_when_input_is_pathlike(self) -> None:
        # Test if input is a file name
        proto = _simple_tensor()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = pathlib.Path(temp_dir, "model.onnx")
            onnx.save_tensor(proto, model_path, format=self.format)
            loaded_proto = onnx.load_tensor(model_path, format=self.format)
            self.assertEqual(proto, loaded_proto)


class TestSaveAndLoadFileExtensions(unittest.TestCase):
    def test_save_model_picks_correct_format_from_extension(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.textproto")
            # No format is specified, so the extension should be used to determine the format
            onnx.save_model(proto, model_path)
            loaded_proto = onnx.load_model(model_path, format="textproto")
            self.assertEqual(proto, loaded_proto)

    def test_load_model_picks_correct_format_from_extension(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.textproto")
            onnx.save_model(proto, model_path, format="textproto")
            # No format is specified, so the extension should be used to determine the format
            loaded_proto = onnx.load_model(model_path)
            self.assertEqual(proto, loaded_proto)

    def test_save_model_uses_format_when_it_is_specified(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.textproto")
            # `format` is specified. It should take precedence over the extension
            onnx.save_model(proto, model_path, format="protobuf")
            loaded_proto = onnx.load_model(model_path, format="protobuf")
            self.assertEqual(proto, loaded_proto)
            with self.assertRaises(google.protobuf.text_format.ParseError):
                # Loading it as textproto (by file extension) should fail
                onnx.load_model(model_path)

    def test_load_model_uses_format_when_it_is_specified(self) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.protobuf")
            onnx.save_model(proto, model_path)
            with self.assertRaises(google.protobuf.text_format.ParseError):
                # `format` is specified. It should take precedence over the extension
                # Loading it as textproto should fail
                onnx.load_model(model_path, format="textproto")

            loaded_proto = onnx.load_model(model_path, format="protobuf")
            self.assertEqual(proto, loaded_proto)

    def test_load_and_save_model_to_path_without_specifying_extension_succeeds(
        self,
    ) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            # No extension is specified
            model_path = os.path.join(temp_dir, "model")
            onnx.save_model(proto, model_path, format="textproto")
            with self.assertRaises(google.protobuf.message.DecodeError):
                # `format` is not specified. load_model should assume protobuf
                # and fail to load it
                onnx.load_model(model_path)

            loaded_proto = onnx.load_model(model_path, format="textproto")
            self.assertEqual(proto, loaded_proto)

    def test_load_and_save_model_without_specifying_extension_or_format_defaults_to_protobuf(
        self,
    ) -> None:
        proto = _simple_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            # No extension is specified
            model_path = os.path.join(temp_dir, "model")
            onnx.save_model(proto, model_path)
            with self.assertRaises(google.protobuf.text_format.ParseError):
                # The model is saved as protobuf, so loading it as textproto should fail
                onnx.load_model(model_path, format="textproto")

            loaded_proto = onnx.load_model(model_path)
            self.assertEqual(proto, loaded_proto)
            loaded_proto_as_explicitly_protobuf = onnx.load_model(
                model_path, format="protobuf"
            )
            self.assertEqual(proto, loaded_proto_as_explicitly_protobuf)


class TestBasicFunctions(unittest.TestCase):
    def test_protos_exist(self) -> None:
        # The proto classes should exist
        _ = onnx.AttributeProto
        _ = onnx.NodeProto
        _ = onnx.GraphProto
        _ = onnx.ModelProto

    def test_version_exists(self) -> None:
        model = onnx.ModelProto()
        # When we create it, graph should not have a version string.
        self.assertFalse(model.HasField("ir_version"))
        # We should touch the version so it is annotated with the current
        # ir version of the running ONNX
        model.ir_version = onnx.IR_VERSION
        model_string = model.SerializeToString()
        model.ParseFromString(model_string)
        self.assertTrue(model.HasField("ir_version"))
        # Check if the version is correct.
        self.assertEqual(model.ir_version, onnx.IR_VERSION)


if __name__ == "__main__":
    unittest.main()
