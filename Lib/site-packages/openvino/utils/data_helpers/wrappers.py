# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from functools import singledispatchmethod
from collections.abc import Mapping
from typing import Dict, Set, Tuple, Union, Iterator, Optional
from typing import KeysView, ItemsView, ValuesView

from openvino._pyopenvino import Tensor, ConstOutput
from openvino._pyopenvino import InferRequest as InferRequestBase


def tensor_from_file(path: str) -> Tensor:
    """Create Tensor from file. Data will be read with dtype of unit8."""
    return Tensor(np.fromfile(path, dtype=np.uint8))  # type: ignore


class _InferRequestWrapper(InferRequestBase):
    """InferRequest class with internal memory."""

    def __init__(self, other: InferRequestBase) -> None:
        # Private memeber to store newly created shared memory data
        self._inputs_data = None
        super().__init__(other)

    def _is_single_input(self) -> bool:
        return len(self.input_tensors) == 1


class OVDict(Mapping):
    """Custom OpenVINO dictionary with inference results.

    This class is a dict-like object. It provides possibility to
    address data tensors with three key types:

    * `openvino.ConstOutput` - port of the output
    * `int` - index of the output
    * `str` - names of the output

    This class follows `frozenset`/`tuple` concept of immutability.
    It is prohibited to assign new items or edit them.

    To revert to the previous behavior use `to_dict` method which
    return shallow copy of underlaying dictionary.
    Note: It removes addressing feature! New dictionary keeps
          only `ConstOutput` keys.

    If a tuple returns value is needed, use `to_tuple` method which
    converts values to the tuple.

    :Example:

    .. code-block:: python

        # Reverts to the previous behavior of the native dict
        result = request.infer(inputs).to_dict()
        # or alternatively:
        result = dict(request.infer(inputs))

    .. code-block:: python

        # To dispatch outputs of multi-ouput inference:
        out1, out2, out3, _ = request.infer(inputs).values()
        # or alternatively:
        out1, out2, out3, _ = request.infer(inputs).to_tuple()
    """
    def __init__(self, _dict: Dict[ConstOutput, np.ndarray]) -> None:
        self._dict = _dict
        self._names: Optional[Dict[ConstOutput, Set[str]]] = None

    def __iter__(self) -> Iterator:
        return self._dict.__iter__()

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return self._dict.__repr__()

    def __get_names(self) -> Dict[ConstOutput, Set[str]]:
        """Return names of every output key.

        Insert empty set if key has no name.
        """
        return {key: key.get_names() for key in self._dict.keys()}

    def __get_key(self, index: int) -> ConstOutput:
        return list(self._dict.keys())[index]

    @singledispatchmethod
    def __getitem_impl(self, key: Union[ConstOutput, int, str]) -> np.ndarray:
        raise TypeError(f"Unknown key type: {type(key)}")

    @__getitem_impl.register
    def _(self, key: ConstOutput) -> np.ndarray:
        return self._dict[key]

    @__getitem_impl.register
    def _(self, key: int) -> np.ndarray:
        try:
            return self._dict[self.__get_key(key)]
        except IndexError:
            raise KeyError(key)

    @__getitem_impl.register
    def _(self, key: str) -> np.ndarray:
        if self._names is None:
            self._names = self.__get_names()
        for port, port_names in self._names.items():
            if key in port_names:
                return self._dict[port]
        raise KeyError(key)

    def __getitem__(self, key: Union[ConstOutput, int, str]) -> np.ndarray:
        return self.__getitem_impl(key)

    def keys(self) -> KeysView[ConstOutput]:
        return self._dict.keys()

    def values(self) -> ValuesView[np.ndarray]:
        return self._dict.values()

    def items(self) -> ItemsView[ConstOutput, np.ndarray]:
        return self._dict.items()

    def names(self) -> Tuple[Set[str], ...]:
        """Return names of every output key.

        Insert empty set if key has no name.
        """
        if self._names is None:
            self._names = self.__get_names()
        return tuple(self._names.values())

    def to_dict(self) -> Dict[ConstOutput, np.ndarray]:
        """Return underlaying native dictionary.

        Function performs shallow copy, thus any modifications to
        returned values may affect this class as well.
        """
        return self._dict

    def to_tuple(self) -> tuple:
        """Convert values of this dictionary to a tuple."""
        return tuple(self._dict.values())
