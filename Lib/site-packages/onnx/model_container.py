# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Implements function make_large_model to easily create and save models
bigger than 2 Gb.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Iterable

import numpy as np

import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker


def _set_external_data(
    tensor: onnx.TensorProto,
    location: str,
    offset: int | None = None,
    length: int | None = None,
    checksum: str | None = None,
    basepath: str | None = None,
) -> None:
    del tensor.external_data[:]
    tensor.data_location = onnx.TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": offset,
        "length": length,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def _enumerate_subgraphs(graph):
    for node in graph.node:
        for att in node.attribute:
            if att.g:
                yield att.g
                yield from _enumerate_subgraphs(att.g)


def make_large_tensor_proto(
    location: str, tensor_name: str, tensor_type: int, shape: tuple[int, ...]
) -> onnx.TensorProto:
    """Create an external tensor.

    Arguments:
        location: unique identifier (not necessary a path)
        tensor_name: tensor name in the graph
        tensor_type: onnx type
        shape: shape the of the initializer

    Returns:
        the created tensor
    """
    tensor_location = location
    tensor = onnx.TensorProto()
    tensor.name = tensor_name
    _set_external_data(tensor, tensor_location)
    tensor.data_type = tensor_type
    tensor.dims.extend(shape)
    return tensor


class ModelContainer:
    """Implements an API to store large tensors outside the main ModelProto,
    it avoids copying large initializers when defining the model and these initializers
    are never serialized through protobuf.
    No tensor is stored on disk until the user explicitly saves the model.
    """

    def __init__(self):
        self.model_proto_: onnx.ModelProto | None = None
        self.large_initializers: dict[str, np.ndarray] = {}

    def check_model(self):
        if self.model_proto is not None:
            onnx.checker.check_model(self.model_proto)

    def __getitem__(self, name: str) -> np.ndarray:
        """Returns an external tensor given its name."""
        if name not in self.large_initializers:
            raise ValueError(
                f"Unable to find large tensor {name!r} among {sorted(self.large_initializers)}."
            )
        return self.large_initializers[name]

    @property
    def model_proto(self) -> onnx.ModelProto:
        if self.model_proto_ is None:
            raise RuntimeError("ModelContainer is empty.")
        return self.model_proto_

    @model_proto.setter
    def model_proto(self, model_proto: onnx.ModelProto):
        self.model_proto_ = model_proto
        self.graphs_ = list(self.enumerate_graph_protos())

    def enumerate_graph_protos(self) -> Iterable[onnx.GraphProto]:
        """Enumerates all GraphProtos in a model."""
        yield self.model_proto.graph
        yield from _enumerate_subgraphs(self.model_proto.graph)

    def is_in_memory_external_initializer(self, name: str) -> bool:
        """Tells if an initializer name is an external initializer stored in memory.
        The name must start with '#' in that case.
        """
        return name.startswith("#")

    def set_large_initializers(self, large_initializers: dict[str, np.ndarray]):
        """Adds all large tensors (not stored in the model)."""
        for k in large_initializers:
            if not self.is_in_memory_external_initializer(k):
                raise ValueError(
                    f"The location {k!r} must start with '#' to be ignored by check model."
                )
        self.large_initializers = large_initializers

    def check_large_initializers(self):
        for tensor in ext_data._get_all_tensors(self.model_proto):
            if not ext_data.uses_external_data(tensor):
                continue
            prop: onnx.StringStringEntryProto | None = None
            for ext in tensor.external_data:  # type: ignore[assignment]
                if ext.key == "location":  # type: ignore[attr-defined]
                    prop = ext
            if prop is None:
                raise RuntimeError(
                    f"No location found for tensor name {tensor.name!r}."
                )
            if prop.value not in self.large_initializers:
                raise RuntimeError(
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )

    def _save_external(
        self, file_path: str, all_tensors_to_one_file: bool
    ) -> onnx.ModelProto:
        """Save the large model into a main onnx file and one file
        per tensor. Follows the same format as :func:`write_external_data_tensors
        <onnx.external_data_helper.write_external_data_tensors>`.
        The main model needs to be modified to update the file location,
        the function returns this modified copy.

        Arguments:
            file_path: model file
            all_tensors_to_one_file: all tensors in one file

        Returns:
            modified main model proto
        """

        def _clean_name(prefix: str, name: str, unique_names: dict[str, int]) -> str:
            if prefix:
                name = f"{prefix}-{name}"
            for c in ":/\\;,!":
                name = name.replace(c, "")
            base_name = name
            if name in unique_names:
                i = unique_names[name] + 1
                unique_names[name] = i
                return f"{base_name}_{i}"
            unique_names[name] = 1
            return name

        unique_names: dict[str, int] = {}
        folder = os.path.dirname(file_path)
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder {folder!r} does not exist.")
        proto = self.model_proto.SerializeToString()
        copy = onnx.ModelProto()
        copy.ParseFromString(proto)
        prefix = os.path.splitext(os.path.split(file_path)[-1])[0]

        if all_tensors_to_one_file:
            file_weight = f"{os.path.split(file_path)[1]}.weight"
            full_file_weight = f"{file_path}.weight"
            offset = 0
            with open(full_file_weight, "wb") as f:
                pass

        for tensor in ext_data._get_all_tensors(copy):
            if not ext_data.uses_external_data(tensor):
                continue
            prop: onnx.StringStringEntryProto | None = None
            for ext in tensor.external_data:  # type: ignore[assignment]
                if ext.key == "location":  # type: ignore[attr-defined]
                    prop = ext  # type: ignore[assignment]
            if prop is None:
                raise RuntimeError(
                    f"No location found for tensor name {tensor.name!r}."
                )
            if prop.value not in self.large_initializers:
                raise RuntimeError(
                    f"Unable to find large tensor named {tensor.name!r} "
                    f"with location {prop.value!r} in "
                    f"{sorted(self.large_initializers)}."
                )
            np_tensor = self.large_initializers[prop.value]

            if sys.byteorder == "big":
                # Convert endian from little to big
                tensor_bytes = np_tensor.byteswap().tobytes()
            else:
                tensor_bytes = np_tensor.tobytes()
            if all_tensors_to_one_file:
                _set_external_data(
                    tensor,
                    location=file_weight,
                    offset=offset,
                    length=len(tensor_bytes),
                )
                offset += len(tensor_bytes)
                with open(full_file_weight, "ab") as f:
                    f.write(tensor_bytes)
            else:
                name = f"{_clean_name(prefix, prop.value, unique_names)}.weight"
                _set_external_data(tensor, location=name)
                full_name = os.path.join(folder, name)
                prop.value = name
                with open(full_name, "wb") as f:
                    f.write(tensor_bytes)

        with open(file_path, "wb") as f:
            f.write(copy.SerializeToString())
        return copy

    def save(
        self,
        file_path: str,
        all_tensors_to_one_file: bool = False,
    ) -> onnx.ModelProto:
        """Save the large model.
        The function returns a ModelProto,
        the current one if the model did not need any modification,
        a modified copy of it if it required changes such as giving file names
        to every external tensor.

        Arguments:
            file_path: model file
            all_tensors_to_one_file: saves all large tensors in one file or
                one file per lerge tensor

        Returns:
            the saved ModelProto
        """
        return self._save_external(
            file_path, all_tensors_to_one_file=all_tensors_to_one_file
        )

    def load(self, file_path: str, load_large_initializers: bool = True):
        """Load the large model.

        Arguments:
            file_path: model file
            load_large_initializers: loads the large initializers,
                if not done, the model is incomplete but it can be used to
                look into the model without executing it and method
                :meth:`_load_large_initializers` can be used to load them later
        """
        self.model_proto_ = onnx.load_model(file_path, load_external_data=False)
        if load_large_initializers:
            self._load_large_initializers(file_path)

    def _load_large_initializers(self, file_path):
        """Loads large initializers.

        Arguments:
            file_path: model file, the weight are expected to be in the same folder as this file
        """
        if self.model_proto_ is None:
            raise RuntimeError("A model must be loaded before loading the weights.")
        self.large_initializers = {}
        base_dir = os.path.dirname(file_path)
        for i, tensor in enumerate(ext_data._get_all_tensors(self.model_proto_)):
            if not ext_data.uses_external_data(tensor):
                continue

            info = ext_data.ExternalDataInfo(tensor)
            external_data_file_path = c_checker._resolve_external_data_location(  # type: ignore[attr-defined]
                base_dir, info.location, tensor.name
            )
            key = f"#t{i}"
            _set_external_data(tensor, location=key)

            with open(external_data_file_path, "rb") as data_file:
                if info.offset:
                    data_file.seek(info.offset)

                raw_data = (
                    data_file.read(info.length) if info.length else data_file.read()
                )

                dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
                shape = tuple(tensor.dims)

                if sys.byteorder == "big":
                    np_tensor = (
                        np.frombuffer(raw_data, dtype=dtype).byteswap().reshape(shape)
                    )
                else:
                    np_tensor = np.frombuffer(raw_data, dtype=dtype).reshape(shape)

                self.large_initializers[key] = np_tensor


def make_large_model(
    graph: onnx.GraphProto,
    large_initializers: dict[str, np.ndarray] | None = None,
    **kwargs: Any,
) -> ModelContainer:
    """Construct a ModelContainer

    C API and Python API of protobuf do not operate without serializing
    the protos. This function uses the Python API of ModelContainer.

    Arguments:
        graph: *make_graph* returns
        large_initializers: dictionary `name: large tensor`,
            large tensor is any python object supporting the DLPack protocol,
            the ownership the tensor is transferred to the ModelContainer,
            the tensor must define method `tobytes` like numpy tensors
        **kwargs: any attribute to add to the returned instance

    Returns:
        ModelContainer
    """
    model = onnx.helper.make_model(graph, **kwargs)
    large_model = ModelContainer()
    large_model.model_proto = model
    if large_initializers:
        large_model.set_large_initializers(large_initializers)
        large_model.check_large_initializers()
    return large_model
