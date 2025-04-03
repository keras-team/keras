# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

__all__ = [
    "registry",
]

import typing
from typing import Any, Collection, Optional, Protocol, TypeVar

import google.protobuf.json_format
import google.protobuf.message
import google.protobuf.text_format

import onnx

_Proto = TypeVar("_Proto", bound=google.protobuf.message.Message)
# Encoding used for serializing and deserializing text files
_ENCODING = "utf-8"


class ProtoSerializer(Protocol):
    """A serializer-deserializer to and from in-memory Protocol Buffers representations."""

    # Format supported by the serializer. E.g. "protobuf"
    supported_format: str
    # File extensions supported by the serializer. E.g. frozenset({".onnx", ".pb"})
    # Be careful to include the dot in the file extension.
    file_extensions: Collection[str]

    # NOTE: The methods defined are serialize_proto and deserialize_proto and not the
    # more generic serialize and deserialize to leave space for future protocols
    # that are defined to serialize/deserialize the ONNX in memory IR.
    # This way a class can implement both protocols.

    def serialize_proto(self, proto: _Proto) -> Any:
        """Serialize a in-memory proto to a serialized data type."""

    def deserialize_proto(self, serialized: Any, proto: _Proto) -> _Proto:
        """Parse a serialized data type into a in-memory proto."""


class _Registry:
    def __init__(self) -> None:
        self._serializers: dict[str, ProtoSerializer] = {}
        # A mapping from file extension to format
        self._extension_to_format: dict[str, str] = {}

    def register(self, serializer: ProtoSerializer) -> None:
        self._serializers[serializer.supported_format] = serializer
        self._extension_to_format.update(
            {ext: serializer.supported_format for ext in serializer.file_extensions}
        )

    def get(self, fmt: str) -> ProtoSerializer:
        """Get a serializer for a format.

        Args:
            fmt: The format to get a serializer for.

        Returns:
            ProtoSerializer: The serializer for the format.

        Raises:
            ValueError: If the format is not supported.
        """
        try:
            return self._serializers[fmt]
        except KeyError:
            raise ValueError(
                f"Unsupported format: '{fmt}'. Supported formats are: {self._serializers.keys()}"
            ) from None

    def get_format_from_file_extension(self, file_extension: str) -> str | None:
        """Get the corresponding format from a file extension.

        Args:
            file_extension: The file extension to get a format for.

        Returns:
            The format for the file extension, or None if not found.
        """
        return self._extension_to_format.get(file_extension)


class _ProtobufSerializer(ProtoSerializer):
    """Serialize and deserialize protobuf message."""

    supported_format = "protobuf"
    file_extensions = frozenset({".onnx", ".pb"})

    def serialize_proto(self, proto: _Proto) -> bytes:
        if hasattr(proto, "SerializeToString") and callable(proto.SerializeToString):
            try:
                result = proto.SerializeToString()
            except ValueError as e:
                if proto.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
                    raise ValueError(
                        "The proto size is larger than the 2 GB limit. "
                        "Please use save_as_external_data to save tensors separately from the model file."
                    ) from e
                raise
            return result  # type: ignore
        raise TypeError(
            f"No SerializeToString method is detected.\ntype is {type(proto)}"
        )

    def deserialize_proto(self, serialized: bytes, proto: _Proto) -> _Proto:
        if not isinstance(serialized, bytes):
            raise TypeError(
                f"Parameter 'serialized' must be bytes, but got type: {type(serialized)}"
            )
        decoded = typing.cast(Optional[int], proto.ParseFromString(serialized))
        if decoded is not None and decoded != len(serialized):
            raise google.protobuf.message.DecodeError(
                f"Protobuf decoding consumed too few bytes: {decoded} out of {len(serialized)}"
            )
        return proto


class _TextProtoSerializer(ProtoSerializer):
    """Serialize and deserialize text proto."""

    supported_format = "textproto"
    file_extensions = frozenset({".textproto", ".prototxt", ".pbtxt"})

    def serialize_proto(self, proto: _Proto) -> bytes:
        textproto = google.protobuf.text_format.MessageToString(proto)
        return textproto.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(
                f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}"
            )
        if isinstance(serialized, bytes):
            serialized = serialized.decode(_ENCODING)
        assert isinstance(serialized, str)
        return google.protobuf.text_format.Parse(serialized, proto)


class _JsonSerializer(ProtoSerializer):
    """Serialize and deserialize JSON."""

    supported_format = "json"
    file_extensions = frozenset({".json", ".onnxjson"})

    def serialize_proto(self, proto: _Proto) -> bytes:
        json_message = google.protobuf.json_format.MessageToJson(
            proto, preserving_proto_field_name=True
        )
        return json_message.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(
                f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}"
            )
        if isinstance(serialized, bytes):
            serialized = serialized.decode(_ENCODING)
        assert isinstance(serialized, str)
        return google.protobuf.json_format.Parse(serialized, proto)


class _TextualSerializer(ProtoSerializer):
    """Serialize and deserialize the ONNX textual representation."""

    supported_format = "onnxtxt"
    file_extensions = frozenset({".onnxtxt", ".onnxtext"})

    def serialize_proto(self, proto: _Proto) -> bytes:
        text = onnx.printer.to_text(proto)  # type: ignore[arg-type]
        return text.encode(_ENCODING)

    def deserialize_proto(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        warnings.warn(
            "The onnxtxt format is experimental. Please report any errors to the ONNX GitHub repository.",
            stacklevel=2,
        )
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(
                f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}"
            )
        if isinstance(serialized, bytes):
            text = serialized.decode(_ENCODING)
        else:
            text = serialized
        if isinstance(proto, onnx.ModelProto):
            return onnx.parser.parse_model(text)  # type: ignore[return-value]
        if isinstance(proto, onnx.GraphProto):
            return onnx.parser.parse_graph(text)  # type: ignore[return-value]
        if isinstance(proto, onnx.FunctionProto):
            return onnx.parser.parse_function(text)  # type: ignore[return-value]
        if isinstance(proto, onnx.NodeProto):
            return onnx.parser.parse_node(text)  # type: ignore[return-value]
        raise ValueError(f"Unsupported proto type: {type(proto)}")


# Register default serializers
registry = _Registry()
registry.register(_ProtobufSerializer())
registry.register(_TextProtoSerializer())
registry.register(_JsonSerializer())
registry.register(_TextualSerializer())
