# Copyright 2020 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TensorStore is a library for reading and writing multi-dimensional arrays."""

import abc as _abc
import builtins as _builtins
import collections.abc as _collections_abc
import typing as _typing

from ._tensorstore import *
from ._tensorstore import _Decodable

newaxis = None
"""Alias for `None` used in :ref:`indexing expressions<python-indexing>` to specify a new singleton dimension.

Example:

    >>> transform = ts.IndexTransform(input_rank=3)
    >>> transform[ts.newaxis, 5]
    Rank 3 -> 3 index space transform:
      Input domain:
        0: [0*, 1*)
        1: (-inf*, +inf*)
        2: (-inf*, +inf*)
      Output index maps:
        out[0] = 5
        out[1] = 0 + 1 * in[1]
        out[2] = 0 + 1 * in[2]

Group:
  Indexing
"""

inf: int
"""Special constant equal to :math:`2^{62}-1` that indicates an unbounded :ref:`index domain<index-domain>`.

Example:

    >>> d = ts.Dim()
    >>> d.inclusive_min
    -4611686018427387903
    >>> d.inclusive_max
    4611686018427387903
    >>> assert d.inclusive_min == -ts.inf
    >>> assert d.inclusive_max == +ts.inf

Group:
  Indexing
"""


class Indexable(metaclass=_abc.ABCMeta):
  """Abstract base class for types that support :ref:`TensorStore indexing operations<python-indexing>`.

  Supported types are:

  - :py:class:`tensorstore.TensorStore`
  - :py:class:`tensorstore.Spec`
  - :py:class:`tensorstore.IndexTransform`

  Group:
    Indexing
  """


Indexable.register(TensorStore)
Indexable.register(Spec)
Indexable.register(IndexTransform)


class FutureLike(metaclass=_abc.ABCMeta):
  """Abstract base class for types representing an asynchronous result.

  The following types may be used where a :py:obj:`FutureLike[T]<.FutureLike>`
  value is expected:

  - an immediate value of type :python:`T`;
  - :py:class:`tensorstore.Future` that resolves to a value of type :python:`T`;
  - :ref:`coroutine<async>` that resolves to a value of type :python:`T`.

  Group:
    Asynchronous support
  """


FutureLike.register(Future)
FutureLike.register(_collections_abc.Coroutine)

bool: dtype
"""Boolean data type (0 or 1).  Corresponds to the :py:obj:`python:bool` type and ``numpy.bool_``.

Group:
  Data types
"""

int4: dtype
"""4-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type, internally stored as its 8-bit signed integer equivalent (i.e. sign-extended). Corresponds to ``jax.numpy.int4``.

Group:
  Data types
"""

int8: dtype
"""8-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int8``.

Group:
  Data types
"""

uint8: dtype
"""8-bit unsigned integer.  Corresponds to ``numpy.uint8``.

Group:
  Data types
"""

int16: dtype
"""16-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int16``.

Group:
  Data types
"""

uint16: dtype
"""16-bit unsigned integer.  Corresponds to ``numpy.uint16``.

Group:
  Data types
"""

int32: dtype
"""32-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int32``.

Group:
  Data types
"""

uint32: dtype
"""32-bit unsigned integer.  Corresponds to ``numpy.uint32``.

Group:
  Data types
"""

int64: dtype
"""32-bit signed :wikipedia:`two's-complement <Two%27s_complement>` integer data type.  Corresponds to ``numpy.int64``.

Group:
  Data types
"""

uint64: dtype
"""64-bit unsigned integer data type.  Corresponds to ``numpy.uint64``.

Group:
  Data types
"""

float8_e4m3fn: dtype
"""8-bit floating-point data type.

Details in https://github.com/jax-ml/ml_dtypes#float8_e4m3fn

Group:
  Data types
"""

float8_e4m3fnuz: dtype
"""8-bit floating-point data type.

Details in https://github.com/jax-ml/ml_dtypes#float8_e4m3fnuz

Group:
  Data types
"""

float8_e4m3b11fnuz: dtype
"""8-bit floating-point data type.

Details in https://github.com/jax-ml/ml_dtypes#float8_e4m3b11fnuz

Group:
  Data types
"""

float8_e5m2: dtype
"""8-bit floating-point data type.

Details in https://github.com/jax-ml/ml_dtypes#float8_e5m2

Group:
  Data types
"""

float8_e5m2fnuz: dtype
"""8-bit floating-point data type.

Details in https://github.com/jax-ml/ml_dtypes#float8_e5m2fnuz

Group:
  Data types
"""

float16: dtype
""":wikipedia:`IEEE 754 binary16 <Half-precision_floating-point_format>` half-precision floating-point data type.  Correspond to ``numpy.float16``.

Group:
  Data types
"""

bfloat16: dtype
""":wikipedia:`bfloat16 floating-point <Bfloat16_floating-point_format>` data type.

NumPy does not have built-in support for bfloat16.  As an extension, TensorStore
defines the :python:`tensorstore.bfloat16.dtype` NumPy data type (also available
as :python:`numpy.dtype("bfloat16")`, as well as the corresponding
:python:`tensorstore.bfloat16.type` :ref:`array scalar
type<numpy:arrays.scalars>`, and these types are guaranteed to interoperate with
`TensorFlow <tensorflow.org>`_ and `JAX <https://github.com/google/jax>`_.

Group:
  Data types
"""

float32: dtype
""":wikipedia:`IEEE 754 binary32 <Single-precision_floating-point_format>` single-precision floating-point data type.  Corresponds to ``numpy.float32``.

Group:
  Data types
"""

float64: dtype
""":wikipedia:`IEEE 754 binary64 <Double-precision_floating-point_format>` double-precision floating-point data type.  Corresponds to ``numpy.float64``.

Group:
  Data types
"""

complex64: dtype
"""Complex number based on :py:obj:`.float32`.  Corresponds to ``numpy.complex64``.

Group:
  Data types
"""

complex128: dtype
"""Complex number based on :py:obj:`.float64`.  Corresponds to ``numpy.complex128``.

Group:
  Data types
"""

string: dtype
"""Variable-length byte string data type.  Corresponds to the Python :py:obj:`python:bytes` type.

There is no precisely corresponding NumPy data type, but ``numpy.object_`` is used.

.. note::

   The :ref:`NumPy string types<numpy:string-dtype-note>`, while related, differ
   in that they are fixed-length and null-terminated.

Group:
  Data types
"""

ustring: dtype
"""Variable-length Unicode string data type.  Corresponds to the Python :py:obj:`python:str` type.

There is no precisely corresponding NumPy data type, but ``numpy.object_`` is used.

.. note::

   The :ref:`NumPy string types<numpy:string-dtype-note>`, while related, differ
   in that they are fixed-length and null-terminated.

Group:
  Data types
"""

json: dtype
"""JSON data type.  Corresponds to an arbitrary Python JSON value.

There is no precisely corresponding NumPy data type, but ``numpy.object_`` is used.

Group:
  Data types
"""

RecheckCacheOption = _typing.Union[
    _builtins.bool, _typing.Literal["open"], _builtins.float
]
"""Determines under what circumstances cached data is revalidated.

``True``
  Revalidate cached data at every option.

``False``
  Assume cached data is always fresh and never revalidate.

``"open"``
  Revalidate cached data older than the time at which the TensorStore was
  opened.

:py:obj:`float`
  Revalidate cached data older than the specified time in seconds since
  the unix epoch.

Group:
  Spec
"""

del _abc
del _builtins
del _collections_abc
del _typing
